using System;
using RacingBotCup.Racing;
using RacingBotCup.Track;
using RacingBotCup.Vehicle;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEngine;

namespace RacingBotCup.Agent
{
    /// <summary>
    /// Base class for a competitor's driver. Subclass it, override
    /// <see cref="Unity.MLAgents.Agent.CollectObservations"/>, and add whatever ray or camera
    /// sensors you like as components alongside it — ML-Agents collects those automatically.
    ///
    /// This replaces the old `obs_config.json` catalogue. That existed so the evaluation harness
    /// never had to compile a competitor's code, which stopped being a constraint the moment
    /// scoring moved onto each competitor's own machine: everyone runs their own build anyway.
    /// What is left is an API rather than a schema — everything the catalogue used to expose is a
    /// property here, and you are no longer limited to what it happened to offer.
    ///
    /// Steering and throttle are the one thing that stays fixed: two continuous actions in -1..1,
    /// or two discrete branches. Declare either on the BehaviorParameters and this class maps it.
    /// </summary>
    public class RacerAgent : Unity.MLAgents.Agent, IDriver
    {
        /// <summary>Discrete steering bins, used when the behaviour declares discrete actions.</summary>
        static readonly float[] k_SteerBins = { -1f, -0.5f, 0f, 0.5f, 1f };

        /// <summary>Discrete throttle bins: brake, coast, accelerate.</summary>
        static readonly float[] k_ThrottleBins = { -1f, 0f, 1f };

        RaceContext m_Context;
        CarController m_Car;
        BehaviorParameters m_Behavior;
        bool m_ManualStepping;
        int m_StepCounter;

        /// <summary>Raised when ML-Agents starts a new episode. The training arena subscribes.</summary>
        public event Action EpisodeBegun;

        public string DriverName => GetType().Name;

        // ------------------------------------------------------------------
        // Everything a policy might want to observe.
        // All of it is safe to read before the harness has bound the agent.
        // ------------------------------------------------------------------

        /// <summary>The car this agent drives. Speed, slip angle, wheels off track, and so on.</summary>
        public CarController Car => m_Car;

        /// <summary>Geometry of the circuit: centreline, width, curvature, typed sections.</summary>
        public TrackModel Track => m_Context?.Track;

        /// <summary>Where the car sits relative to the centreline right now.</summary>
        public TrackModel.Projection Projection => m_Context?.Projection ?? default;

        public CheckpointRing Checkpoints => m_Context?.Checkpoints;

        /// <summary>Seconds since the last checkpoint was credited.</summary>
        public float TimeSinceCheckpoint => m_Context?.TimeSinceCheckpoint ?? 0f;

        /// <summary>Fraction of the lap completed, 0 to 1.</summary>
        public float LapProgress => m_Context?.Checkpoints.Progress ?? 0f;

        public bool IsOffTrack => m_Context != null && !m_Context.Projection.IsOnRoad;

        /// <summary>True once the harness has wired this agent to a car and a circuit.</summary>
        public bool IsBound => m_Context != null;

        /// <summary>
        /// The lap time this episode is being measured against, in seconds — a reference run's time
        /// on this exact circuit plus a margin. Zero when nothing has set one.
        ///
        /// A reward paid for a lap time needs a target, and the target cannot be a constant: a
        /// practice circuit is anywhere from 800 m to 2000 m long with a different corner mix each
        /// time, so a time that is a good lap on one seed is an impossible one on the next. The
        /// training arena looks the number up per seed and hands it over here before the episode
        /// starts — see <see cref="TrainingArena"/> and <see cref="Track.SeedPool"/>.
        /// </summary>
        public float ReferenceTime { get; private set; }

        public bool HasReferenceTime => ReferenceTime > 0f;

        /// <summary>
        /// A point on the centreline this far ahead, in the car's local space.
        /// Positive Z is in front of the car, positive X is to its right.
        /// </summary>
        public Vector3 WaypointLocal(float metresAhead)
        {
            if (m_Context == null)
            {
                return Vector3.zero;
            }

            var world = m_Context.Track.SampleAtDistance(m_Context.Projection.Distance + metresAhead).Position;
            return transform.InverseTransformPoint(world);
        }

        /// <summary>
        /// Mean signed curvature (1/radius) over a window starting <paramref name="metresAhead"/>
        /// down the circuit. Positive turns one way, negative the other.
        /// </summary>
        public float CurvatureAhead(float metresAhead, float window = 20f)
        {
            return m_Context == null
                ? 0f
                : m_Context.Track.AverageCurvature(m_Context.Projection.Distance + metresAhead, window);
        }

        /// <summary>What kind of section is coming up — straight, hairpin, chicane, and so on.</summary>
        public TrackSection SectionAhead(float metresAhead = 0f)
        {
            return m_Context == null
                ? default
                : m_Context.Track.SectionAt(m_Context.Projection.Distance + metresAhead);
        }

        // ------------------------------------------------------------------
        // Harness wiring. Competitors do not call these.
        // ------------------------------------------------------------------

        /// <summary>Chooses whether the agent steps the Academy itself (evaluation) or is driven
        /// by a DecisionRequester and automatic stepping (training).</summary>
        public void Configure(bool manualStepping)
        {
            m_ManualStepping = manualStepping;
        }

        /// <summary>Sets the target lap time for the episode about to start. Training only.</summary>
        public void SetReferenceTime(float seconds)
        {
            ReferenceTime = Mathf.Max(0f, seconds);
        }

        public void Bind(RaceContext context)
        {
            m_Context = context;
            m_Car = context.Car;
        }

        public void BeginRun()
        {
            m_StepCounter = 0;
        }

        public void EndRun()
        {
            if (m_Car != null)
            {
                m_Car.SetInput(0f, 0f);
            }
        }

        /// <summary>
        /// Asks for a decision on the cadence the policy was trained at
        /// (<see cref="Eval.RaceRules.DecisionPeriod"/>); between decisions the car holds its last
        /// input.
        ///
        /// Deliberately does not step the Academy. Ten environments run at once during evaluation,
        /// and stepping once per agent would advance the Academy ten times per physics tick — the
        /// harness makes that single call after every agent has asked.
        /// </summary>
        public void Tick()
        {
            if (!m_ManualStepping)
            {
                return;
            }

            if (m_StepCounter++ % Eval.RaceRules.DecisionPeriod == 0)
            {
                RequestDecision();
            }
        }

        public override void OnEpisodeBegin()
        {
            EpisodeBegun?.Invoke();
        }

        /// <summary>
        /// Maps the policy's output onto the car. Override <see cref="OnDriveApplied"/> rather than
        /// this if you want to add reward shaping — the mapping itself has to stay put so that every
        /// entry drives the same car the same way.
        /// </summary>
        public override void OnActionReceived(ActionBuffers actions)
        {
            float steer;
            float throttle;

            if (UsesDiscreteActions)
            {
                var discrete = actions.DiscreteActions;
                steer = k_SteerBins[Mathf.Clamp(discrete[0], 0, k_SteerBins.Length - 1)];
                throttle = k_ThrottleBins[Mathf.Clamp(discrete[1], 0, k_ThrottleBins.Length - 1)];
            }
            else
            {
                var continuous = actions.ContinuousActions;
                steer = Mathf.Clamp(continuous[0], -1f, 1f);
                throttle = Mathf.Clamp(continuous[1], -1f, 1f);
            }

            if (m_Car != null)
            {
                m_Car.SetInput(steer, throttle);
            }

            OnDriveApplied(steer, throttle);
        }

        /// <summary>
        /// Called every time an action reaches the car. This is where reward shaping goes.
        ///
        /// 보상은 학습 시점에만 존재하고 채점에 전혀 개입하지 않습니다 (기획서 §3) — so shape it
        /// however you like. <see cref="RacingBotCup.Agent.SampleRacerAgent"/> shows one way.
        /// </summary>
        protected virtual void OnDriveApplied(float steer, float throttle)
        {
        }

        /// <summary>
        /// The training arena calls this when the lap is finished. Override to reward it.
        /// Never called during evaluation — scoring reads lap times, not rewards.
        /// </summary>
        public virtual void OnLapCompleted(float elapsedSeconds)
        {
        }

        /// <summary>The training arena calls this when a run is abandoned (off track, or too slow).</summary>
        public virtual void OnRunFailed()
        {
        }

        /// <summary>Reads the action space off the behaviour rather than making you declare it twice.</summary>
        public bool UsesDiscreteActions
        {
            get
            {
                if (m_Behavior == null)
                {
                    m_Behavior = GetComponent<BehaviorParameters>();
                }

                return m_Behavior != null && m_Behavior.BrainParameters.ActionSpec.NumDiscreteActions > 0;
            }
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var (steer, throttle) = ReadKeyboard();

            if (UsesDiscreteActions)
            {
                var discrete = actionsOut.DiscreteActions;
                discrete[0] = NearestBin(k_SteerBins, steer);
                discrete[1] = NearestBin(k_ThrottleBins, throttle);
            }
            else
            {
                var continuous = actionsOut.ContinuousActions;
                continuous[0] = steer;
                continuous[1] = throttle;
            }
        }

        /// <summary>
        /// Manual driving input for the heuristic policy.
        ///
        /// This project has active input handling set to the Input System package, where touching
        /// the legacy <c>UnityEngine.Input</c> class throws rather than returning zero — so an
        /// evaluation run with no model supplied would fault on every decision step.
        /// </summary>
        static (float Steer, float Throttle) ReadKeyboard()
        {
#if ENABLE_INPUT_SYSTEM
            var keyboard = UnityEngine.InputSystem.Keyboard.current;
            if (keyboard == null)
            {
                return (0f, 0f);
            }

            var steer = Axis(keyboard.dKey.isPressed || keyboard.rightArrowKey.isPressed,
                             keyboard.aKey.isPressed || keyboard.leftArrowKey.isPressed);
            var throttle = Axis(keyboard.wKey.isPressed || keyboard.upArrowKey.isPressed,
                                keyboard.sKey.isPressed || keyboard.downArrowKey.isPressed);
            return (steer, throttle);
#elif ENABLE_LEGACY_INPUT_MANAGER
            return (Input.GetAxis("Horizontal"), Input.GetAxis("Vertical"));
#else
            return (0f, 0f);
#endif
        }

#if ENABLE_INPUT_SYSTEM
        static float Axis(bool positive, bool negative)
        {
            return (positive ? 1f : 0f) - (negative ? 1f : 0f);
        }
#endif

        static int NearestBin(float[] bins, float value)
        {
            var best = 0;
            var bestDistance = float.MaxValue;

            for (var i = 0; i < bins.Length; i++)
            {
                var distance = Mathf.Abs(bins[i] - value);
                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    best = i;
                }
            }

            return best;
        }

        /// <summary>The two action spaces the car accepts, for wiring a BehaviorParameters in code.</summary>
        public static ActionSpec ContinuousActionSpec => ActionSpec.MakeContinuous(2);

        public static ActionSpec DiscreteActionSpec =>
            ActionSpec.MakeDiscrete(k_SteerBins.Length, k_ThrottleBins.Length);
    }
}

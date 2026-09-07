using RacingBotCup.Eval;
using RacingBotCup.Racing;
using RacingBotCup.Track;
using RacingBotCup.Vehicle;
using Unity.MLAgents;
using UnityEngine;

namespace RacingBotCup.Agent
{
    /// <summary>
    /// Runs training episodes on the circuit and car already sitting in the scene.
    ///
    /// It no longer builds anything: the track and the car are scene objects you can select and
    /// inspect before pressing Play. All this does is decide when an episode is over and put the
    /// car back on the start line — plus, by default, roll a new circuit each time.
    ///
    /// Rolling a new circuit every episode is the point of the whole competition. A policy trained
    /// on one layout learns that layout; the score is for generalisation, so the ground has to keep
    /// moving.
    ///
    /// The other thing it does is hand the agent a clock. Ground that keeps moving is also ground
    /// whose lap times are not comparable — 800 m to 2000 m, with a different corner mix on each
    /// one — so an agent paid for going quickly needs to know what quickly means on the circuit it
    /// just drew. <see cref="ApplyReferenceTime"/> looks that up per seed and sets it before the
    /// episode starts, and the same number is what the episode is timed out against.
    ///
    /// It also feeds the clock back. A seed the reference recording retired on trains against an
    /// estimate — see <see cref="Track.SeedPool"/> — until the agent itself clears it, at which
    /// point <see cref="TryFillMissingReferenceTime"/> turns that lap into a real measurement so
    /// every later episode on the same seed gets an actual target instead of a guess.
    /// </summary>
    [DefaultExecutionOrder(-100)]
    public sealed class TrainingArena : MonoBehaviour
    {
        [Header("씬 참조")]
        [SerializeField] TrackInstance m_Track;

        [Tooltip("씬에 놓인 차. 자식에 RacerAgent가 붙어 있어야 합니다")]
        [SerializeField] CarController m_Car;

        [Header("커리큘럼")]
        [Tooltip("에피소드마다 새 트랙을 뽑습니다. 끄면 씬의 시드를 계속 씁니다")]
        [SerializeField] bool m_RandomizeEachEpisode = true;

        [Tooltip("비워두면 아래 대역 전체에서 뽑습니다. 채우면 그 목록에서만 뽑아 특정 종류의 코스만 반복 학습합니다")]
        [SerializeField] SeedPool m_SeedPool;

        [Tooltip("연습용 시드 대역. 평가 시드와 겹치지 않게 두세요")]
        [SerializeField] int m_SeedMin;

        [SerializeField] int m_SeedMax = 100000;

        [Tooltip("어떤 기준 시간을 잡아도 이 시간을 넘으면 에피소드를 끝냅니다")]
        [SerializeField] float m_MaxEpisodeSeconds = 200f;

        [Header("기준 시간")]
        [Tooltip("기록된 기준 랩타임에 얹는 여유(초). 에이전트가 노리는 목표 시간은 '기록 + 이 값'입니다")]
        [SerializeField] float m_ReferenceMargin = 1f;

        [Tooltip("기준 시간의 몇 배가 지나면 리타이어로 처리할지. 목표의 두 배 반을 쓰고도 못 " +
                 "끝냈다면 그 에피소드에서 더 배울 것은 없습니다")]
        [SerializeField] float m_TimeoutMultiplier = 2.5f;

        [Tooltip("기록이 전혀 없을 때 쓰는 기준 속도(초/미터). 시드 풀이 없는 일반 학습 씬과 " +
                 "아직 측정하지 않은 풀이 여기로 떨어집니다")]
        [SerializeField] float m_FallbackPaceSecondsPerMetre = 0.05f;

        /// <summary>
        /// Minimum real seconds between two disk writes for the same pool. Several parallel training
        /// areas can each fill in a different unmeasured seed within the same second; without a
        /// floor here, every one of those would trigger its own full <c>AssetDatabase.SaveAssets()</c>.
        /// Shared by every <see cref="TrainingArena"/> in the scene — keyed by pool, not by arena, so
        /// eight areas pointed at the same pool still only save it this often between them.
        /// </summary>
        const float k_ReferenceTimeSaveIntervalSeconds = 30f;

        static readonly System.Collections.Generic.Dictionary<SeedPool, float> s_LastReferenceTimeSave =
            new System.Collections.Generic.Dictionary<SeedPool, float>();

        RacerAgent m_Agent;
        RaceContext m_Context;
        DeterministicRandom m_Random;
        bool m_Ready;

        void Awake()
        {
            m_Random = new DeterministicRandom(System.Environment.TickCount ^ GetInstanceID());
            Time.fixedDeltaTime = CarSpec.FixedDeltaTime;
        }

        void Start()
        {
            if (!Resolve())
            {
                enabled = false;
                return;
            }

            m_Agent.Configure(manualStepping: false);
            m_Agent.EpisodeBegun += OnEpisodeBegun;

            if (m_Agent.GetComponent<DecisionRequester>() == null)
            {
                var requester = m_Agent.gameObject.AddComponent<DecisionRequester>();
                requester.DecisionPeriod = RaceRules.DecisionPeriod;
                requester.TakeActionsBetweenDecisions = false;
            }

            ApplyReferenceTime();
            ResetCar();
            m_Ready = true;
        }

        bool Resolve()
        {
            if (m_Track == null)
            {
                m_Track = FindFirstObjectByType<TrackInstance>();
            }

            if (m_Car == null)
            {
                m_Car = FindFirstObjectByType<CarController>();
            }

            if (m_Track == null || m_Car == null)
            {
                Debug.LogError("[RacingBotCup] Training scene needs a TrackInstance and a car. " +
                               "Run RacingBotCup > Build Scenes to recreate it.");
                return false;
            }

            m_Agent = m_Car.GetComponentInChildren<RacerAgent>();
            if (m_Agent == null)
            {
                Debug.LogError($"[RacingBotCup] No RacerAgent found under '{m_Car.name}'. " +
                               "Drag your agent prefab in as a child of the car.");
                return false;
            }

            return true;
        }

        void OnDestroy()
        {
            if (m_Agent != null)
            {
                m_Agent.EpisodeBegun -= OnEpisodeBegun;
            }
        }

        void OnEpisodeBegun()
        {
            if (!m_Ready)
            {
                return;
            }

            if (m_RandomizeEachEpisode)
            {
                m_Track.Randomize(NextSeed());
            }

            ApplyReferenceTime();
            ResetCar();
        }

        /// <summary>
        /// Tells the agent what lap time this episode is being measured against, so its reward can
        /// be paid for beating a clock rather than merely for finishing (see
        /// <see cref="SampleRacerAgent.OnLapCompleted"/>). Called after the circuit is rolled,
        /// because the answer depends on which circuit it landed on.
        ///
        /// Three places it can come from, in order of how much they are worth trusting:
        /// <list type="number">
        /// <item>The time <see cref="Eval.BaselineTimeRecorder"/> measured on this exact seed.</item>
        /// <item>The pool's median pace over the seeds it did measure, applied to this circuit's
        /// length — for a seed the reference bot retired on. A drill's hardest circuits are the ones
        /// most likely to end up here and the last ones worth dropping from the curriculum, so they
        /// get a length-adjusted estimate rather than being skipped.</item>
        /// <item><see cref="m_FallbackPaceSecondsPerMetre"/>, for a scene with no pool at all — the
        /// general training scene rerolls the whole practice band and has nothing measured.</item>
        /// </list>
        /// </summary>
        void ApplyReferenceTime()
        {
            m_Agent.SetReferenceTime(ReferenceTimeFor(m_Track.Seed));
        }

        float ReferenceTimeFor(int seed)
        {
            if (m_SeedPool != null && m_SeedPool.TryGetBaselineSeconds(seed, out var measured))
            {
                return measured + m_ReferenceMargin;
            }

            var pace = m_SeedPool != null && m_SeedPool.MedianPaceSecondsPerMetre > 0f
                ? m_SeedPool.MedianPaceSecondsPerMetre
                : m_FallbackPaceSecondsPerMetre;

            return pace * m_Track.Model.TotalLength + m_ReferenceMargin;
        }

        /// <summary>
        /// Turns an estimate into a measurement. A seed the reference bot retired on has no real
        /// time behind its target — <see cref="ReferenceTimeFor"/> is guessing from the pool's
        /// median pace — right up until training itself proves the seed clearable, at which point
        /// there is no reason to keep guessing. This is that handoff: the lap just completed is
        /// offered to <see cref="SeedPool.TryRecordTrainingTime"/>, which only accepts it if the
        /// seed still has no official measurement.
        ///
        /// Only mutates the pool's in-memory copy, cheaply, on every qualifying lap — the actual
        /// disk write is throttled (see <see cref="k_ReferenceTimeSaveIntervalSeconds"/>) because
        /// several parallel training areas can each finish a lap within the same second, and
        /// <c>AssetDatabase.SaveAssets()</c> is a full asset-database write, not something to call
        /// that often.
        /// </summary>
        void TryFillMissingReferenceTime()
        {
            if (m_SeedPool == null)
            {
                return;
            }

            // The same correction evaluation and the reference recorder apply, backing off the part
            // of the final step taken after the line — without it a training-filled time would run
            // slightly long next to one BaselineTimeRecorder measured, for no reason but which tool
            // produced it.
            var lapSeconds = m_Context.ElapsedTime -
                              (1f - m_Context.Checkpoints.LapCrossingFraction) * Time.fixedDeltaTime;

            if (!m_SeedPool.TryRecordTrainingTime(m_Track.Seed, lapSeconds))
            {
                return;
            }

#if UNITY_EDITOR
            UnityEditor.EditorUtility.SetDirty(m_SeedPool);

            var now = Time.realtimeSinceStartup;
            if (!s_LastReferenceTimeSave.TryGetValue(m_SeedPool, out var last) ||
                now - last >= k_ReferenceTimeSaveIntervalSeconds)
            {
                UnityEditor.AssetDatabase.SaveAssets();
                s_LastReferenceTimeSave[m_SeedPool] = now;
            }
#endif
        }

        /// <summary>
        /// The circuit for the next episode: one drawn from the seed pool when there is one,
        /// otherwise anything in the practice band. A pool is how you train against a feature that
        /// only a fraction of random seeds happen to contain — see <see cref="SeedPool"/>.
        /// </summary>
        int NextSeed()
        {
            if (m_SeedPool != null && m_SeedPool.Count > 0)
            {
                return m_SeedPool.Next(ref m_Random);
            }

            return m_Random.Range(m_SeedMin, Mathf.Max(m_SeedMin + 1, m_SeedMax));
        }

        void ResetCar()
        {
            var rig = new RacerRig(m_Car, m_Agent);
            m_Context = rig.PlaceOnTrack(m_Track.Model);
            Physics.SyncTransforms();
        }

        void FixedUpdate()
        {
            if (!m_Ready || m_Context == null)
            {
                return;
            }

            m_Context.Refresh(Time.fixedDeltaTime);

            if (m_Context.LapCompletedThisTick)
            {
                RecordEpisodeStats(lapCompleted: true);
                TryFillMissingReferenceTime();
                m_Agent.OnLapCompleted(m_Context.ElapsedTime);
                m_Agent.EndEpisode();
                return;
            }

            if (m_Context.OffTrackDuration >= RaceRules.OffTrackDnfSeconds ||
                m_Context.ElapsedTime >= EpisodeDeadline)
            {
                RecordEpisodeStats(lapCompleted: false);
                m_Agent.OnRunFailed();
                m_Agent.EndEpisode();
            }
        }

        /// <summary>
        /// When to give up on the current episode. A multiple of the target time rather than a flat
        /// ceiling: the point of the clock is to be measured against the reference, and a car that
        /// has spent two and a half times the target still going is not going to produce a lap worth
        /// learning from — ending it early buys another episode instead. The flat ceiling stays as a
        /// backstop for the case where no target could be worked out at all.
        /// </summary>
        float EpisodeDeadline
        {
            get
            {
                var reference = m_Agent.ReferenceTime;
                return reference > 0f
                    ? Mathf.Min(m_MaxEpisodeSeconds, reference * Mathf.Max(1f, m_TimeoutMultiplier))
                    : m_MaxEpisodeSeconds;
            }
        }

        /// <summary>
        /// Reports the final state of every training episode to TensorBoard.
        ///
        /// The clear time is only logged when the lap actually finished. Logging every episode's
        /// elapsed time mixed real lap times in with runs that went off track after two seconds or
        /// sat at the timeout ceiling, so the curve tracked how the car failed rather than how fast
        /// it was. Progress and the clear rate still go in on every episode — the clear rate is what
        /// says whether a falling clear time means a faster car or just an easier sample of laps.
        /// </summary>
        void RecordEpisodeStats(bool lapCompleted)
        {
            var stats = Academy.Instance.StatsRecorder;
            stats.Add("Episode/Progress", m_Context.Checkpoints.Progress);
            stats.Add("Episode/ClearRate", lapCompleted ? 1f : 0f);

            if (!lapCompleted)
            {
                return;
            }

            stats.Add("Episode/ClearSeconds", m_Context.ElapsedTime);

            // The one number that is comparable across circuits, and the one the reward is actually
            // paid on: above 1 the lap beat its target, below it did not. Clear time alone cannot
            // say that — a 55 s lap is excellent on a 2 km circuit and slow on an 800 m one, so the
            // curve moves with which seeds happened to come up as much as with how the car drove.
            if (m_Agent.HasReferenceTime && m_Context.ElapsedTime > 0f)
            {
                stats.Add("Episode/ReferenceRatio", m_Agent.ReferenceTime / m_Context.ElapsedTime);
            }
        }
    }
}

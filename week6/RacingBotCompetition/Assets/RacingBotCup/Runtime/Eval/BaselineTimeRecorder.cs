using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using RacingBotCup.Agent;
using RacingBotCup.Racing;
using RacingBotCup.Track;
using RacingBotCup.Vehicle;
using Unity.InferenceEngine;
using Unity.MLAgents;
using UnityEngine;

namespace RacingBotCup.Eval
{
    /// <summary>
    /// Drives a trained model around every circuit in a set of <see cref="SeedPool"/>s and writes
    /// the lap times back into them.
    ///
    /// This is what makes a drill's reward mean something. Paying for a lap time only works against
    /// a target, and a pool's circuits run from 800 m to 2000 m with a different corner mix on each
    /// one, so a single hand-picked target time would be generous on one seed and impossible on the
    /// next — the policy would learn which seeds to bother on rather than how to drive. Measuring
    /// each seed once, up front, turns "go fast" into a per-circuit target the agent can actually be
    /// scored against.
    ///
    /// <b>Which model.</b> Whatever <see cref="m_Model"/> is assigned in the Inspector, run through
    /// <see cref="m_AgentPrefab"/> exactly as it would be driven at evaluation time — this is
    /// <see cref="Racing.RacerBuilder.BuildAgent"/> in inference mode, not the hand-tuned
    /// <see cref="RacingBotCup.Agent.BaselineBot"/> the competition score is measured against. Pick
    /// a checkpoint you trust and keep it fixed for the length of a training run: it is the target
    /// the reward is shaped around, and a target that moves — say, because it gets re-recorded from
    /// whatever the policy currently in training happens to produce — turns the reward into a
    /// moving-goalpost feedback loop rather than a fixed thing to improve against.
    ///
    /// <b>Retirements.</b> Even a good model does not clear every circuit — a bad landing off a
    /// RampCorner, a line that clips a crate. Those seeds are recorded as unmeasured rather than
    /// thrown away, and the pool also stores the median seconds-per-metre of everything that did
    /// finish, so <see cref="Agent.TrainingArena"/> can still hand the agent a length-adjusted target
    /// on them. Losing a quarter of a drill's hardest circuits — precisely the ones worth practising
    /// — to a failed measurement would be the wrong trade.
    ///
    /// Runs in play mode: it steps <c>Physics.Simulate</c> and the ML-Agents <see cref="Academy"/> by
    /// hand, as fast as the editor will go, one circuit at a time. Six drill pools of a hundred seeds
    /// is a few minutes.
    /// </summary>
    [DefaultExecutionOrder(-50)]
    public sealed class BaselineTimeRecorder : MonoBehaviour
    {
        /// <summary>Standing-start settling before the clock starts, matching evaluation.</summary>
        const float k_SettleSeconds = 0.5f;

        [Header("측정 대상")]
        [Tooltip("기준 시간을 기록할 시드 풀들. 각 풀의 모든 시드를 한 번씩 돕니다")]
        [SerializeField] List<SeedPool> m_Pools = new List<SeedPool>();

        [Tooltip("기준으로 삼을 차. 에이전트가 학습에 쓰는 차와 같아야 기록이 의미가 있습니다")]
        [SerializeField] GameObject m_CarPrefab;

        [Header("기준 모델")]
        [Tooltip("기준 모델을 조종할 에이전트 프리팹. RacerAgent를 상속한 컴포넌트가 붙어 있어야 합니다")]
        [SerializeField] GameObject m_AgentPrefab;

        [Tooltip("기준으로 삼을 학습된 .onnx 모델. 학습 중인 정책이 아니라 고정해둔 체크포인트를 " +
                 "쓰세요 — 매번 바뀌는 모델을 기준으로 잡으면 보상이 뒤쫓는 목표가 계속 움직여 " +
                 "학습이 수렴하지 않습니다")]
        [SerializeField] ModelAsset m_Model;

        [Header("설정 (기본값 그대로 두세요)")]
        [SerializeField] TrackMaterials m_Materials = new TrackMaterials();

        [SerializeField] TrackPropCatalogue m_Props = new TrackPropCatalogue();

        [Tooltip("한 바퀴에 허용하는 최대 시간. 이 시간을 넘기면 그 시드는 기록 없음으로 남습니다")]
        [SerializeField] float m_TimeoutSeconds = RaceRules.BaselineTimeoutSeconds;

        [Tooltip("프레임당 물리 스텝 수. 크게 잡을수록 빨리 끝나고 에디터가 그만큼 덜 반응합니다")]
        [SerializeField] int m_StepsPerFrame = 500;

        [Header("실행")]
        [Tooltip("플레이를 누르면 바로 측정을 시작합니다")]
        [SerializeField] bool m_RunOnStart = true;

        [Tooltip("측정이 끝나면 플레이 모드를 빠져나옵니다")]
        [SerializeField] bool m_ExitPlayModeWhenDone = true;

        [Header("진행 상황")]
        [SerializeField] string m_Status = "idle";

        [SerializeField] string m_Summary = "";

        public string Status => m_Status;

        public bool IsRunning { get; private set; }

        void Start()
        {
            if (m_RunOnStart)
            {
                Record();
            }
        }

        /// <summary>Runs the whole measurement. Does nothing if one is already in flight.</summary>
        [ContextMenu("Record now")]
        public void Record()
        {
            if (IsRunning)
            {
                return;
            }

            if (!Application.isPlaying)
            {
                Debug.LogWarning("[RacingBotCup] The reference times are measured in play mode. Press Play.");
                return;
            }

            if (m_AgentPrefab == null || m_Model == null)
            {
                Debug.LogError(
                    "[RacingBotCup] Assign an agent prefab and a trained model before recording " +
                    "reference times — there is nothing to drive without them.");
                return;
            }

            StartCoroutine(RecordRoutine());
        }

        IEnumerator RecordRoutine()
        {
            IsRunning = true;

            var previousFixedDelta = Time.fixedDeltaTime;
            var previousSimulationMode = Physics.simulationMode;
            var previousRunInBackground = Application.runInBackground;
            var previousSkidMarks = SkidMarks.GloballyEnabled;
            var previousAutoStep = !Academy.IsInitialized || Academy.Instance.AutomaticSteppingEnabled;

            Time.fixedDeltaTime = CarSpec.FixedDeltaTime;
            Physics.simulationMode = SimulationMode.Script;
            Application.runInBackground = true;
            Academy.Instance.AutomaticSteppingEnabled = false;

            // Nothing is being watched and every circuit is torn down the moment it is timed, so the
            // marks are pure cost.
            SkidMarks.GloballyEnabled = false;

            var summary = new StringBuilder();

            try
            {
                foreach (var pool in m_Pools)
                {
                    if (pool == null || pool.Count == 0)
                    {
                        continue;
                    }

                    var times = new float[pool.Count];
                    var paces = new List<float>(pool.Count);

                    for (var i = 0; i < pool.Count; i++)
                    {
                        var seed = pool.SeedAt(i);
                        m_Status = $"{pool.name}  {i + 1}/{pool.Count}  (seed {seed})";

                        float seconds = SeedPool.UnmeasuredSeconds;
                        float length = 0f;
                        yield return RunSeed(seed, result =>
                        {
                            seconds = result.Seconds;
                            length = result.TrackLength;
                        });

                        times[i] = seconds;
                        if (seconds > SeedPool.UnmeasuredSeconds && length > 0f)
                        {
                            paces.Add(seconds / length);
                        }
                    }

                    var pace = Median(paces);
                    var note = $"{paces.Count}/{pool.Count} circuits timed with {m_Model.name}, " +
                               $"median pace {pace:F4} s/m, recorded {DateTime.Now:yyyy-MM-dd HH:mm}.";

                    Write(pool, times, pace, note);
                    summary.AppendLine($"{pool.name}: {note}");

                    if (paces.Count < pool.Count)
                    {
                        Debug.LogWarning(
                            $"[RacingBotCup] The reference model retired on {pool.Count - paces.Count} of " +
                            $"{pool.name}'s {pool.Count} circuits. Those seeds keep training, with a " +
                            "target estimated from the pool's median pace and their own length.");
                    }
                }
            }
            finally
            {
                Time.fixedDeltaTime = previousFixedDelta;
                Physics.simulationMode = previousSimulationMode;
                Application.runInBackground = previousRunInBackground;
                SkidMarks.GloballyEnabled = previousSkidMarks;
                if (Academy.IsInitialized)
                {
                    Academy.Instance.AutomaticSteppingEnabled = previousAutoStep;
                }

                IsRunning = false;
            }

            m_Summary = summary.ToString();
            m_Status = "done";
            Debug.Log($"[RacingBotCup] Reference times recorded.{Environment.NewLine}{m_Summary}");

#if UNITY_EDITOR
            if (m_ExitPlayModeWhenDone)
            {
                UnityEditor.EditorApplication.ExitPlaymode();
            }
#endif
        }

        readonly struct SeedResult
        {
            public readonly float Seconds;
            public readonly float TrackLength;

            public SeedResult(float seconds, float trackLength)
            {
                Seconds = seconds;
                TrackLength = trackLength;
            }
        }

        /// <summary>
        /// Builds one circuit, drives the reference model around it, and reports the lap time — or
        /// <see cref="SeedPool.UnmeasuredSeconds"/> if the car went off for
        /// <see cref="RaceRules.OffTrackDnfSeconds"/> or ran out of clock. Everything it built is
        /// destroyed before it returns, so a hundred seeds cost one circuit's worth of memory.
        /// </summary>
        IEnumerator RunSeed(int seed, Action<SeedResult> onDone)
        {
            var root = new GameObject($"Measure_{seed}");
            root.transform.SetParent(transform, false);

            var trackObject = new GameObject("Circuit");
            trackObject.transform.SetParent(root.transform, false);

            var track = trackObject.AddComponent<TrackInstance>();
            track.Seed = seed;

            // Exactly how a drill scene generates it, or the circuit measured here is not the
            // circuit the agent will be timed against.
            track.EnableHazardSections = true;
            CopyMaterials(track);
            CopyProps(track);
            track.Rebuild();

            // manualStepping: true and BehaviorType.InferenceOnly (set automatically because a model
            // is supplied) — the same combination EvaluationRunner drives a competitor's own agent
            // with, so a lap timed here is driven exactly as it would be at evaluation.
            var rig = RacerBuilder.BuildAgent(
                m_CarPrefab, m_AgentPrefab, m_Model, manualStepping: true, parent: root.transform);
            if (rig == null)
            {
                Destroy(root);
                onDone(new SeedResult(SeedPool.UnmeasuredSeconds, 0f));
                yield break;
            }

            var context = rig.PlaceOnTrack(track.Model);
            var dt = CarSpec.FixedDeltaTime;
            var isAgentDriven = rig.Driver is RacerAgent;

            Physics.SyncTransforms();

            for (var i = 0; i < Mathf.CeilToInt(k_SettleSeconds / dt); i++)
            {
                rig.Car.SetInput(0f, 0f);
                rig.Car.Step(dt);
                Physics.Simulate(dt);
            }

            context.Reset();
            rig.Driver.BeginRun();

            var seconds = SeedPool.UnmeasuredSeconds;
            var maxSteps = Mathf.CeilToInt(m_TimeoutSeconds / dt);
            var stepsThisFrame = 0;

            for (var step = 0; step < maxSteps; step++)
            {
                context.Refresh(dt);

                if (context.LapCompletedThisTick)
                {
                    // Back off the part of the final step taken after the line, the same correction
                    // evaluation applies, so a recorded time and a scored time are the same quantity.
                    seconds = context.ElapsedTime - (1f - context.Checkpoints.LapCrossingFraction) * dt;
                    break;
                }

                if (context.OffTrackDuration >= RaceRules.OffTrackDnfSeconds)
                {
                    break;
                }

                rig.Driver.Tick();

                // RacerAgent.Tick() only requests a decision; nothing actually reaches OnActionReceived
                // — and so CarController.SetInput — until the Academy is stepped, exactly as
                // EvaluationRunner steps it for the agents it drives.
                if (isAgentDriven)
                {
                    Academy.Instance.EnvironmentStep();
                }

                rig.Car.Step(dt);
                Physics.Simulate(dt);

                if (++stepsThisFrame >= Mathf.Max(1, m_StepsPerFrame))
                {
                    stepsThisFrame = 0;
                    yield return null;
                }
            }

            var length = track.Model.TotalLength;

            rig.Driver.EndRun();
            Destroy(root);

            // The circuit's colliders survive until the end of the frame, and the next seed's
            // circuit is built at the same origin — overlapping the two would have the next car
            // settle on top of this one's geometry.
            yield return null;

            onDone(new SeedResult(seconds, length));
        }

        void CopyMaterials(TrackInstance track)
        {
            track.Materials.Road = m_Materials.Road;
            track.Materials.Runoff = m_Materials.Runoff;
            track.Materials.Ground = m_Materials.Ground;
        }

        void CopyProps(TrackInstance track)
        {
            track.Props.Barriers = m_Props.Barriers;
            track.Props.Crates = m_Props.Crates;
            track.Props.Logs = m_Props.Logs;
            track.Props.Containers = m_Props.Containers;
            track.Props.Ramp = m_Props.Ramp;
        }

        static float Median(List<float> values)
        {
            if (values.Count == 0)
            {
                return 0f;
            }

            values.Sort();
            var middle = values.Count / 2;
            return values.Count % 2 == 1
                ? values[middle]
                : 0.5f * (values[middle - 1] + values[middle]);
        }

        /// <summary>
        /// Writes the times into the pool asset. Editor-only by necessity — a built player has no
        /// asset database to write to, and nothing in a built player would want to.
        ///
        /// Also clears every seed's <c>m_TrainingFilled</c> flag: a fresh official pass is
        /// authoritative, so nothing a training run filled in between recordings survives it, and
        /// every seed measured here (or left genuinely unmeasured) stops — or starts — being
        /// eligible for <see cref="SeedPool.TryRecordTrainingTime"/> to touch again.
        /// </summary>
        static void Write(SeedPool pool, float[] times, float medianPace, string note)
        {
#if UNITY_EDITOR
            var serialized = new UnityEditor.SerializedObject(pool);

            var array = serialized.FindProperty("m_BaselineSeconds");
            array.arraySize = times.Length;
            for (var i = 0; i < times.Length; i++)
            {
                array.GetArrayElementAtIndex(i).floatValue = times[i];
            }

            var filled = serialized.FindProperty("m_TrainingFilled");
            filled.arraySize = times.Length;
            for (var i = 0; i < times.Length; i++)
            {
                filled.GetArrayElementAtIndex(i).boolValue = false;
            }

            serialized.FindProperty("m_MedianPaceSecondsPerMetre").floatValue = medianPace;
            serialized.FindProperty("m_BaselineNote").stringValue = note;
            serialized.ApplyModifiedPropertiesWithoutUndo();

            UnityEditor.EditorUtility.SetDirty(pool);
            UnityEditor.AssetDatabase.SaveAssets();
#else
            Debug.LogWarning($"[RacingBotCup] {pool.name} was measured but cannot be written outside the editor.");
#endif
        }
    }
}

using Unity.MLAgents.Sensors;
using UnityEngine;

namespace RacingBotCup.Agent
{
    /// <summary>
    /// A worked example of a competition entry. Copy this file, rename the class, and change
    /// whatever you like — this is a starting point, not a recommendation.
    ///
    /// It shows the two things you are expected to design (기획서 §3):
    /// <list type="number">
    /// <item><b>Observations</b> — what the policy is allowed to see, in
    /// <see cref="CollectObservations"/> plus whatever sensor components you add in the Inspector.</item>
    /// <item><b>Rewards</b> — what counts as doing well, in <see cref="OnDriveApplied"/> and the
    /// two episode hooks. Rewards exist only while training and never touch your score.</item>
    /// </list>
    ///
    /// Everything read below comes from <see cref="RacerAgent"/>. Values are scaled to roughly
    /// -1..1 by hand here; a policy learns much faster when its inputs are on a similar scale, and
    /// raw metres per second next to a 0-to-1 lap fraction is not.
    ///
    /// <b>What an episode is worth.</b> The whole return is built around one number — the lap time,
    /// measured against a reference run on the same circuit (see <see cref="OnLapCompleted"/>).
    /// Everything else is small enough to stay out of its way:
    ///
    /// <list type="bullet">
    /// <item><b>Retired</b> — off the road too long, or still going at a multiple of the target time:
    /// <c>-1</c>, plus whatever progress it had made, so a car that crashes out at the last corner
    /// still lands under a car that finished.</item>
    /// <item><b>Finished</b> — <c>1</c> at exactly the target time, climbing without any ceiling the
    /// quicker it goes — <c>ratio 1.65 ≈ 1.5</c>, <c>ratio 2.0 ≈ 2.5</c>, and onward, so an
    /// exceptional lap is never worth the same as a merely-fast one — and decaying towards zero the
    /// slower, but never to zero and never below it. Finishing therefore always beats retiring,
    /// however bad the lap was.</item>
    /// <item><b>Along the way</b> — up to <c>+0.3</c> of lap progress, a fraction of a point for
    /// sitting off the racing surface, and about <c>0.15</c> a crash. Enough to shape the early
    /// episodes, when a finish is still rare; not enough to out-earn the finish once they are not.</item>
    /// </list>
    ///
    /// Sized deliberately against <c>gamma</c> in the trainer config: a terminal reward this dominant
    /// only teaches anything if it survives the discount back to the start of a 700-decision lap.
    /// </summary>
    public class SampleRacerAgent : RacerAgent
    {
        [Header("Observation shape")]
        [Tooltip("How many centreline points ahead to look at.")]
        [SerializeField] int m_Waypoints = 5;

        [Tooltip("Metres between those points.")]
        [SerializeField] float m_WaypointSpacing = 10f;

        [Tooltip("How many curvature windows ahead to sample.")]
        [SerializeField] int m_CurvatureWindows = 3;

        [SerializeField] float m_CurvatureWindowLength = 20f;

        [Header("Finishing — what a lap is worth")]
        [Tooltip("정확히 기준 시간에 완주했을 때의 보상. 이보다 빠르면 올라가고 느리면 내려갑니다")]
        [SerializeField] float m_OnReferenceReward = 1f;

        [Tooltip("기준 시간 대비 얼마나 민감하게 반응할지. 크면 1초 단축의 가치가 커집니다")]
        [SerializeField] float m_TimeExponent = 4f;

        [Tooltip("기준보다 빠를 때 보상이 커지는 배율. ratio=2일 때 보상은 대략 " +
                 "'m_OnReferenceReward + 이 값'이 됩니다 (ratio-1이 정확히 1이므로)")]
        [SerializeField] float m_OvershootScale = 1.5f;

        [Tooltip("기준보다 빠를 때 커브가 얼마나 볼록한지. 클수록 목표 근처에서는 완만하다가 " +
                 "더 빨라질수록 보상이 급격히 커집니다. 상한이 없으므로 이 값이 실질적인 " +
                 "민감도를 결정합니다")]
        [SerializeField] float m_OvershootExponent = 2.55f;

        [Tooltip("리타이어했을 때의 처벌. 아무리 느려도 완주는 항상 양수이므로, 완주는 언제나 리타이어보다 낫습니다")]
        [SerializeField] float m_FailurePenalty = 1f;

        [Header("Reward shaping")]
        [Tooltip("한 바퀴를 다 돌았을 때 누적되는 전진 보상의 총량. 실제로는 진행한 비율만큼 나눠 지급됩니다")]
        [SerializeField] float m_ProgressReward = 0.3f;

        [Tooltip("Charged every decision while any part of the car is off the racing surface.")]
        [SerializeField] float m_OffTrackPenalty = 0.003f;

        [Header("Collision penalty")]
        [Tooltip("Charged when the car loses speed far faster than the brakes or the grass could take it — i.e. it hit something. Scales up to double this on a full-speed hit.")]
        [SerializeField] float m_ImpactPenalty = 0.15f;

        [Tooltip("Deceleration (m/s^2) above which a speed loss counts as an impact. Full braking peaks near 25 and four wheels on the grass near 15, so keep this above both.")]
        [SerializeField] float m_ImpactDecelThreshold = 35f;

        [Tooltip("A throttle at or below this is the brake pedal, and losing speed with it pressed is intentional rather than a crash.")]
        [SerializeField] float m_BrakeThreshold = -0.05f;

        [Tooltip("Below this speed (m/s) a knock is not worth charging for.")]
        [SerializeField] float m_ImpactMinSpeed = 3f;

        [Tooltip("Seconds of quiet before another impact can be charged, so one crash spread over several decisions is billed once.")]
        [SerializeField] float m_ImpactCooldown = 0.5f;

        float m_LastProgress;
        float m_LastSpeed;
        float m_LastSpeedTime;
        float m_ImpactCooldownRemaining;

        /// <summary>
        /// Total floats written below. Put this number in BehaviorParameters →
        /// Vector Observation → Space Size, or the policy and the observations disagree and
        /// ML-Agents throws on the first decision.
        /// </summary>
        public int ObservationSize => 8 + m_Waypoints * 2 + m_CurvatureWindows;

        public override void CollectObservations(VectorSensor sensor)
        {
            if (!IsBound)
            {
                // Bound by the harness before the first decision; this guard only matters if you
                // drop the agent into a scene by hand.
                for (var i = 0; i < ObservationSize; i++)
                {
                    sensor.AddObservation(0f);
                }

                return;
            }

            // --- how the car is moving (5 floats) ---
            sensor.AddObservation(Car.ForwardSpeed / 50f);
            sensor.AddObservation(Car.LocalVelocity.x / 50f);
            sensor.AddObservation(Car.LocalAngularVelocity.y / 5f);
            sensor.AddObservation(Car.SteerAngleNormalized);
            sensor.AddObservation(Car.SlipAngle / 45f);

            // --- where it is on the circuit (3 floats) ---
            var projection = Projection;
            var halfWidth = Mathf.Max(0.5f, projection.Width * 0.5f);
            sensor.AddObservation(projection.Lateral / halfWidth);   // ±1 at the road edges
            sensor.AddObservation(projection.Width / 12f);
            sensor.AddObservation(IsOffTrack ? 1f : 0f);

            // --- what is coming up (m_Waypoints * 2 + m_CurvatureWindows floats) ---
            for (var i = 1; i <= m_Waypoints; i++)
            {
                var local = WaypointLocal(i * m_WaypointSpacing);
                sensor.AddObservation(local.x / 50f);
                sensor.AddObservation(local.z / 50f);
            }

            for (var i = 0; i < m_CurvatureWindows; i++)
            {
                var curvature = CurvatureAhead(i * m_CurvatureWindowLength, m_CurvatureWindowLength);
                sensor.AddObservation(Mathf.Clamp(curvature * 30f, -3f, 3f));
            }
        }

        // ------------------------------------------------------------------
        // 보상 설계 — 여기서부터가 여러분의 영역입니다.
        // ------------------------------------------------------------------

        protected override void OnDriveApplied(float steer, float throttle)
        {
            if (!IsBound)
            {
                return;
            }

            // Progress along the centreline, as a fraction of the lap rather than in metres, so the
            // same drive down an 800 m circuit and a 2000 m one is paid the same — and so the whole
            // term is worth m_ProgressReward over a completed lap no matter which seed came up.
            // Measured as a delta: the agent is paid for advancing, not for being far along, and the
            // episode total telescopes to exactly how much of the lap it got round.
            //
            // Nothing here charges for time any more. Time is priced once, at the finish line,
            // against this circuit's own reference — and pricing it twice would put the two in
            // competition and make the range of an episode's return impossible to reason about.
            // Crawling is still the worst thing the car can do: the arena retires it at a multiple
            // of the reference time, which is the -m_FailurePenalty it was trying to avoid.
            var progress = Checkpoints.Progress;
            AddReward((progress - m_LastProgress) * m_ProgressReward);
            m_LastProgress = progress;

            if (IsOffTrack)
            {
                AddReward(-m_OffTrackPenalty);
            }

            ChargeForImpact(throttle);
        }

        /// <summary>
        /// Charges for hitting something — an obstacle on an ObstacleStraight, a prop off the side of
        /// the road. The car exposes no collision callback, so an impact is inferred from the
        /// telemetry instead: speed collapsing far faster than anything the driver asked for.
        ///
        /// Priced at a sixth of a lap on target, which is roughly what a crash costs in time anyway.
        /// It used to be three times what finishing a lap was worth, from when the lap bonus was 20;
        /// on this scale that would make one unlucky graze worse than not finishing at all, and the
        /// safest policy available would be to stop.
        ///
        /// Two things keep honest driving out of it. Braking is excluded by the throttle sign, and the
        /// threshold sits above what the car can shed on its own — full braking is roughly 25 m/s^2
        /// and the off-road drag another 15, while running into something stationary is an order of
        /// magnitude more. Speed magnitude rather than forward speed is what is watched, so a spin,
        /// which rotates the velocity without destroying it, does not read as a crash.
        /// </summary>
        void ChargeForImpact(float throttle)
        {
            var speed = Car.Speed;
            var now = Time.fixedTime;
            var deltaTime = now - m_LastSpeedTime;
            var previousSpeed = m_LastSpeed;

            m_LastSpeed = speed;
            m_LastSpeedTime = now;
            m_ImpactCooldownRemaining = Mathf.Max(0f, m_ImpactCooldownRemaining - Mathf.Max(0f, deltaTime));

            // Evaluation steps physics by hand and never advances Time.fixedTime, which would make
            // the rate meaningless. Rewards do not exist there anyway.
            if (deltaTime <= 0f)
            {
                return;
            }

            if (m_ImpactCooldownRemaining > 0f ||
                previousSpeed < m_ImpactMinSpeed ||
                throttle <= m_BrakeThreshold)
            {
                return;
            }

            var deceleration = (previousSpeed - speed) / deltaTime;
            if (deceleration < m_ImpactDecelThreshold)
            {
                return;
            }

            // Twice the threshold or worse is a solid hit rather than a graze; charge double there.
            var severity = Mathf.Clamp01((deceleration - m_ImpactDecelThreshold) / m_ImpactDecelThreshold);
            AddReward(-m_ImpactPenalty * (1f + severity));
            m_ImpactCooldownRemaining = m_ImpactCooldown;
        }

        /// <summary>
        /// Pays for the lap against the clock it was set to beat. Two different curves either side
        /// of the target, both anchored at <c>(ratio=1, m_OnReferenceReward)</c>:
        /// <list type="bullet">
        /// <item>Below target (<c>ratio ≤ 1</c>): <c>ratio^m_TimeExponent</c> — decays smoothly to
        /// zero as the lap gets slower, so a finish is still worth something however far off the
        /// pace, and never negative.</item>
        /// <item>Above target (<c>ratio &gt; 1</c>): <c>m_OnReferenceReward + m_OvershootScale ·
        /// (ratio − 1)^m_OvershootExponent</c> — climbs without limit. There is deliberately no
        /// ceiling here.</item>
        /// </list>
        ///
        /// The reference is a real time — <see cref="Eval.BaselineTimeRecorder"/> drove a trusted
        /// checkpoint round this exact seed and <see cref="TrainingArena"/> handed the result over
        /// with a second of margin on it. That per-seed target is the whole point. A flat "finishing
        /// is worth 20, minus a bit per second" pays a 2000 m circuit far less than an 800 m one for
        /// exactly the same quality of driving, so the policy learns to read the reward as which
        /// seed it drew rather than as how it drove, and the gradient that is left points at
        /// finishing rather than at being quick.
        ///
        /// <b>History.</b> The first version clipped at a flat ceiling with <c>Mathf.Min</c>; a live
        /// run showed <c>Episode/ReferenceRatio</c> sitting around 1.6–1.75 for the length of
        /// training, well past where the clip kicked in, so the large majority of finishes landed on
        /// the exact same number. Swapping the clip for a smooth asymptote fixed the collision but
        /// still packed most finishes into a narrow band under the ceiling. The default constants
        /// below were solved to hit specific, explicitly chosen points instead —
        /// <c>ratio 1.65 → ≈1.5</c>, <c>ratio 2.0 → ≈2.5</c> — with nothing above them capping the
        /// climb.
        ///
        /// It also stays positive however slow the lap was, so finishing badly still beats retiring
        /// — <see cref="OnRunFailed"/> is negative and this never is.
        /// </summary>
        public override void OnLapCompleted(float elapsedSeconds)
        {
            AddReward(FinishReward(elapsedSeconds));
        }

        float FinishReward(float elapsedSeconds)
        {
            // No reference to measure against — the arena always supplies one, so this is the
            // hand-placed-in-a-scene case. Pay a finished lap what hitting the target is worth and
            // leave the time out of it, rather than inventing a target the run was never set.
            if (!HasReferenceTime || elapsedSeconds <= 0f)
            {
                return m_OnReferenceReward;
            }

            var ratio = ReferenceTime / elapsedSeconds;

            if (ratio <= 1f)
            {
                return m_OnReferenceReward * Mathf.Pow(ratio, m_TimeExponent);
            }

            return m_OnReferenceReward + m_OvershootScale * Mathf.Pow(ratio - 1f, m_OvershootExponent);
        }

        public override void OnRunFailed()
        {
            AddReward(-m_FailurePenalty);
        }

        public override void OnEpisodeBegin()
        {
            m_LastProgress = 0f;
            m_LastSpeed = 0f;
            m_LastSpeedTime = Time.fixedTime;
            m_ImpactCooldownRemaining = 0f;
            base.OnEpisodeBegin();
        }

    }
}

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

        [Header("Reward shaping")]
        [Tooltip("Per metre of forward progress along the circuit.")]
        [SerializeField] float m_ProgressReward = 0.02f;

        [Tooltip("Charged every decision, so standing still is never comfortable.")]
        [SerializeField] float m_TimePenalty = 0.001f;

        [Tooltip("Charged while any part of the car is off the racing surface.")]
        [SerializeField] float m_OffTrackPenalty = 0.01f;

        [SerializeField] float m_LapBonus = 20f;

        [SerializeField] float m_FailurePenalty = 3f;

        [Header("Collision penalty")]
        [Tooltip("Charged when the car loses speed far faster than the brakes or the grass could take it — i.e. it hit something. Scales up to double this on a full-speed hit.")]
        [SerializeField] float m_ImpactPenalty = 3f;

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

            // Progress along the centreline is the primary signal. Measured as a delta so the agent
            // is paid for advancing, not for being far along.
            var progress = Checkpoints.TraveledDistance;
            AddReward((progress - m_LastProgress) * m_ProgressReward);
            m_LastProgress = progress;

            AddReward(-m_TimePenalty);

            if (IsOffTrack)
            {
                AddReward(-m_OffTrackPenalty);
            }

            ChargeForImpact(throttle);
        }

        /// <summary>
        /// Charges hard for hitting something — an obstacle on an ObstacleStraight, a prop off the
        /// side of the road. The car exposes no collision callback, so an impact is inferred from the
        /// telemetry instead: speed collapsing far faster than anything the driver asked for.
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

        public override void OnLapCompleted(float elapsedSeconds)
        {
            // Finishing is worth a lot, finishing quickly a little more.
            AddReward(Mathf.Max(1f, m_LapBonus - elapsedSeconds * 0.05f));
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

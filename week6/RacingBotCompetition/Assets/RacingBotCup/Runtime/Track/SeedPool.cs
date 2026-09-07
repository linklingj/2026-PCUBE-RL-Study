using System;
using System.Collections.Generic;
using UnityEngine;

namespace RacingBotCup.Track
{
    /// <summary>
    /// A fixed shortlist of circuits to train on, in place of the whole practice band — together
    /// with a reference lap time for each of them.
    ///
    /// <see cref="Agent.TrainingArena"/> normally rerolls a seed at random every episode, which is
    /// exactly right for general driving: the score is for generalisation, so the ground has to keep
    /// moving. It is the wrong sampler for a feature that only turns up on a quarter of circuits —
    /// hazard sections appear with 25% probability each (see
    /// <see cref="CircuitLayout.RollHazards"/>), so three episodes in four teach the policy nothing
    /// about them. Pointing an arena at a pool built from seeds that all contain the feature turns
    /// that quarter into all of it.
    ///
    /// The pool is a plain list of seeds rather than a filter evaluated at runtime because
    /// generating a circuit to find out whether it qualifies costs milliseconds, and rejecting three
    /// out of four of them at the start of every episode would spend a noticeable slice of training
    /// on tracks that are thrown away. Build one from the <c>RacingBotCup → Drill Scenes</c> menu,
    /// which writes a pool per kind of section; the seeds are reproducible, so the same pool comes
    /// back on every machine.
    ///
    /// <b>Reference times.</b> A drill's reward is paid for beating a clock rather than for merely
    /// finishing (see <see cref="Agent.SampleRacerAgent"/>), and every circuit in the pool is a
    /// different length with a different corner mix, so one shared target time would be meaningless:
    /// the same lap time is a good run on one seed and a bad one on the next. So the pool carries a
    /// measured time per seed, recorded by driving <see cref="Agent.BaselineBot"/> around every one
    /// of them — <c>RacingBotCup → Drill Scenes → Record Reference Times</c>.
    ///
    /// A seed the reference bot retired on has no time, and is stored as
    /// <see cref="UnmeasuredSeconds"/> rather than dropped: the seed is still worth drilling, it
    /// just needs a target from somewhere else. <see cref="MedianPaceSecondsPerMetre"/> is that
    /// somewhere — the median seconds-per-metre over the seeds that did finish, which turned into a
    /// target for an unmeasured seed is a length-adjusted estimate rather than a guess.
    ///
    /// <b>Filling the gaps from training.</b> An estimate is not a measurement, and the seeds it
    /// covers are exactly the ones a reference run failed on — often the hardest circuits in the
    /// pool, and so the ones most worth having a real number for. The moment training itself clears
    /// one of them, <see cref="Agent.TrainingArena"/> hands the lap time to
    /// <see cref="TryRecordTrainingTime"/>, which promotes that seed from estimated to measured.
    /// A seed filled this way keeps tightening — a faster training lap later replaces a slower one —
    /// right up until <c>Record Reference Times</c> is run again, which always wins outright: an
    /// official measurement is the deliberately chosen target for the whole run, and no training
    /// episode gets to move it once it exists.
    /// </summary>
    [CreateAssetMenu(menuName = "RacingBotCup/Seed Pool", fileName = "seed_pool")]
    public sealed class SeedPool : ScriptableObject
    {
        /// <summary>
        /// What <see cref="BaselineSecondsAt"/> returns for a seed with no recorded time — either
        /// because the reference bot retired on it or because the pool has never been measured.
        /// </summary>
        public const float UnmeasuredSeconds = 0f;

        [Tooltip("이 묶음이 무슨 기준으로 뽑혔는지. 생성 도구가 채웁니다")]
        [SerializeField] string m_Description = "";

        [Tooltip("학습에 쓸 시드 목록. 매 에피소드 여기서만 하나를 뽑습니다")]
        [SerializeField] int[] m_Seeds = Array.Empty<int>();

        [Tooltip("시드별 기준 랩타임(초). m_Seeds와 같은 순서이며, 0은 기준 봇이 그 시드에서 " +
                 "완주하지 못해 기록이 없다는 뜻입니다")]
        [SerializeField] float[] m_BaselineSeconds = Array.Empty<float>();

        [Tooltip("완주한 시드들의 초/미터 중앙값. 기록이 없는 시드의 기준 시간을 트랙 길이에 " +
                 "맞춰 추정할 때 씁니다")]
        [SerializeField] float m_MedianPaceSecondsPerMetre;

        [Tooltip("기준 시간을 언제, 어떻게 측정했는지. 기록 도구가 채웁니다")]
        [SerializeField] string m_BaselineNote = "";

        [Tooltip("m_BaselineSeconds의 각 값이 훈련 중 자동으로 채워진 것인지 표시합니다. " +
                 "true인 항목만 훈련이 더 빠른 기록으로 덮어쓸 수 있고, false(공식 측정)는 " +
                 "Record Reference Times를 다시 돌리기 전까지 훈련이 건드리지 않습니다")]
        [SerializeField] bool[] m_TrainingFilled = Array.Empty<bool>();

        public string Description => m_Description;

        public IReadOnlyList<int> Seeds => m_Seeds;

        public int Count => m_Seeds == null ? 0 : m_Seeds.Length;

        public string BaselineNote => m_BaselineNote;

        /// <summary>
        /// Median pace over the seeds the reference bot finished, in seconds per metre. Zero until
        /// the pool has been measured.
        /// </summary>
        public float MedianPaceSecondsPerMetre => m_MedianPaceSecondsPerMetre;

        /// <summary>How many seeds in the pool carry a measured reference time.</summary>
        public int MeasuredCount
        {
            get
            {
                if (m_BaselineSeconds == null)
                {
                    return 0;
                }

                var measured = 0;
                foreach (var seconds in m_BaselineSeconds)
                {
                    if (seconds > UnmeasuredSeconds)
                    {
                        measured++;
                    }
                }

                return measured;
            }
        }

        /// <summary>The seed at <paramref name="index"/>, wrapped so any index is safe.</summary>
        public int SeedAt(int index)
        {
            return Count == 0 ? 0 : m_Seeds[((index % Count) + Count) % Count];
        }

        /// <summary>
        /// The reference lap time recorded for the seed at <paramref name="index"/>, or
        /// <see cref="UnmeasuredSeconds"/> if there is none. Wrapped like <see cref="SeedAt"/>.
        /// </summary>
        public float BaselineSecondsAt(int index)
        {
            if (Count == 0 || m_BaselineSeconds == null || m_BaselineSeconds.Length != Count)
            {
                return UnmeasuredSeconds;
            }

            return m_BaselineSeconds[((index % Count) + Count) % Count];
        }

        /// <summary>
        /// The reference lap time for a particular seed. False when the seed is not in this pool, or
        /// is in it but the reference bot never set a time on it — the caller has to fall back to
        /// <see cref="MedianPaceSecondsPerMetre"/> either way, so both are one answer.
        /// </summary>
        public bool TryGetBaselineSeconds(int seed, out float seconds)
        {
            seconds = UnmeasuredSeconds;

            if (Count == 0 || m_BaselineSeconds == null || m_BaselineSeconds.Length != Count)
            {
                return false;
            }

            for (var i = 0; i < m_Seeds.Length; i++)
            {
                if (m_Seeds[i] != seed)
                {
                    continue;
                }

                seconds = m_BaselineSeconds[i];
                return seconds > UnmeasuredSeconds;
            }

            return false;
        }

        /// <summary>Draws one seed. Takes the stream by reference so the caller's sequence advances.</summary>
        public int Next(ref DeterministicRandom random)
        {
            return Count == 0 ? 0 : m_Seeds[random.Range(0, Count)];
        }

        /// <summary>
        /// Fills in this seed's reference time from a lap actually completed during training, or
        /// tightens one already filled this way if <paramref name="seconds"/> is faster. Does
        /// nothing to a seed that already carries an official measurement — that number is the
        /// deliberately chosen target for the run, and a training episode has no business moving it.
        ///
        /// Mutates the pool in memory only; persisting the change to disk (so it survives past this
        /// Editor session) is the caller's job — see <see cref="Agent.TrainingArena"/>.
        /// </summary>
        /// <returns>Whether anything actually changed, so the caller knows whether the update is
        /// worth persisting.</returns>
        public bool TryRecordTrainingTime(int seed, float seconds)
        {
            if (Count == 0 || seconds <= UnmeasuredSeconds)
            {
                return false;
            }

            if (m_BaselineSeconds == null || m_BaselineSeconds.Length != Count)
            {
                m_BaselineSeconds = new float[Count];
            }

            if (m_TrainingFilled == null || m_TrainingFilled.Length != Count)
            {
                m_TrainingFilled = new bool[Count];
            }

            for (var i = 0; i < m_Seeds.Length; i++)
            {
                if (m_Seeds[i] != seed)
                {
                    continue;
                }

                var current = m_BaselineSeconds[i];

                if (current > UnmeasuredSeconds && !m_TrainingFilled[i])
                {
                    return false; // officially measured — hands off until re-recorded
                }

                if (current > UnmeasuredSeconds && seconds >= current)
                {
                    return false; // already have a training-filled time, and this one isn't faster
                }

                m_BaselineSeconds[i] = seconds;
                m_TrainingFilled[i] = true;
                return true;
            }

            return false;
        }
    }
}

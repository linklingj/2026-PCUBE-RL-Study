using System.Collections.Generic;
using System.IO;
using RacingBotCup.Track;
using UnityEditor;
using UnityEngine;

namespace RacingBotCup.EditorTools
{
    /// <summary>
    /// Finds the circuits that contain a particular kind of section and stores them as a
    /// <see cref="SeedPool"/> for a training arena to draw from.
    ///
    /// Hazard sections are rolled independently at 25% per seed (see
    /// <see cref="CircuitLayout.RollHazards"/>), so a policy training on random practice seeds meets
    /// an <see cref="TrackSectionType.ObstacleStraight"/> in about one episode in four — thin
    /// repetition for a skill as specific as threading a line between scattered crates. Doing the
    /// search once, up front, is what turns that quarter into all of it.
    ///
    /// <see cref="DrillSceneBootstrap"/> is what calls this, once per kind of section it builds a
    /// scene for.
    ///
    /// The scan walks seeds in order from a starting point rather than sampling them at random, so
    /// the pool is reproducible: the same seeds come out on every machine, and re-running the tool
    /// rewrites the asset with what it already contained instead of reshuffling the curriculum
    /// underneath a half-finished training run.
    /// </summary>
    public static class SeedPoolBuilder
    {
        public const string ConfigDirectory = "Assets/RacingBotCup/Config";

        /// <summary>
        /// Seeds examined between progress-bar repaints. Generating a circuit takes a few
        /// milliseconds, so repainting on every one of them costs more than the scan does.
        /// </summary>
        const int k_ProgressInterval = 25;

        /// <summary>
        /// Collects up to <paramref name="count"/> seeds whose circuit contains at least one of
        /// <paramref name="required"/>, looking at no more than <paramref name="maxScanned"/> seeds.
        /// Returns however many it found — a short list means the scan ran out of budget (or the
        /// user cancelled), not that the caller should retry.
        /// </summary>
        /// <param name="required">
        /// Section types that qualify a circuit, matched as "any of". One entry drills a single kind
        /// of section; several make a mixed pool.
        /// </param>
        /// <param name="label">What to call the filter in the progress bar, e.g. "a RampCorner".</param>
        public static List<int> FindSeeds(
            IReadOnlyList<TrackSectionType> required,
            int count,
            int firstSeed,
            int maxScanned,
            string label)
        {
            var found = new List<int>(count);

            try
            {
                for (var offset = 0; offset < maxScanned && found.Count < count; offset++)
                {
                    var seed = firstSeed + offset;

                    if (offset % k_ProgressInterval == 0 &&
                        EditorUtility.DisplayCancelableProgressBar(
                            $"Scanning for circuits with {label}",
                            $"{found.Count} / {count} found — at seed {seed}",
                            found.Count / (float)count))
                    {
                        break;
                    }

                    // Hazards on, exactly as TrackInstance generates them, so a seed that qualifies
                    // here produces the same circuit in the scene.
                    var track = TrackGenerator.Generate(seed, enableHazardSections: true);
                    if (ContainsAny(track.Model, required))
                    {
                        found.Add(seed);
                    }
                }
            }
            finally
            {
                EditorUtility.ClearProgressBar();
            }

            return found;
        }

        static bool ContainsAny(TrackModel model, IReadOnlyList<TrackSectionType> types)
        {
            foreach (var section in model.Sections)
            {
                for (var i = 0; i < types.Count; i++)
                {
                    if (section.Type == types[i])
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        /// <summary>Creates the pool asset, or overwrites the one already at <paramref name="path"/>
        /// so that every scene referencing it picks up the new seeds.</summary>
        public static SeedPool WriteAsset(string path, IReadOnlyList<int> seeds, string description)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path));

            var pool = AssetDatabase.LoadAssetAtPath<SeedPool>(path);
            if (pool == null)
            {
                pool = ScriptableObject.CreateInstance<SeedPool>();
                AssetDatabase.CreateAsset(pool, path);
            }

            var serialized = new SerializedObject(pool);
            serialized.FindProperty("m_Description").stringValue = description;

            var array = serialized.FindProperty("m_Seeds");
            array.arraySize = seeds.Count;
            for (var i = 0; i < seeds.Count; i++)
            {
                array.GetArrayElementAtIndex(i).intValue = seeds[i];
            }

            serialized.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(pool);
            AssetDatabase.SaveAssets();

            return pool;
        }
    }
}

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using RacingBotCup.Agent;
using RacingBotCup.Eval;
using RacingBotCup.Track;
using RacingBotCup.Vehicle;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace RacingBotCup.EditorTools
{
    /// <summary>
    /// Builds a training scene per kind of section — one that only ever rolls circuits with an
    /// obstacle straight on them, one for sharp hairpins, one for ramp corners, and so on.
    ///
    /// There is a scene per kind rather than one scene with a dropdown because a drill is a training
    /// run, not a setting. Each is a separate run id with its own reward curve to read, and
    /// switching between them by editing a field leaves the scene dirty in git every time you change
    /// your mind about what to practise. Six scenes cost nothing: everything except the seed pool is
    /// identical, down to the four parallel environments, the behaviour name and the agent prefab,
    /// so any of them trains with the same config and the same command.
    ///
    /// Drills exist for the sections a random practice seed rarely produces. Straight, Corner and
    /// Hairpin are on effectively every circuit already — the layout grammar seeds every one with a
    /// main straight into a hairpin (see <see cref="CircuitLayout.Build"/>) — so a pool filtered on
    /// those would be every seed, which is what the general training scene already gives you. What
    /// is left is the three hazard variants at 25% each, plus chicanes and esses, which are common
    /// but not guaranteed.
    ///
    /// None of these replaces the general scene. A policy that only ever sees obstacle straights
    /// will weave beautifully and forget how to carry speed down a clear one; the intended use is to
    /// fine-tune from a general checkpoint and then go back.
    /// </summary>
    public static class DrillSceneBootstrap
    {
        public const string SceneDirectory = SceneBootstrap.SceneDirectory + "/Drills";

        public const string SeedDirectory = SeedPoolBuilder.ConfigDirectory + "/DrillSeeds";

        /// <summary>Where <see cref="BuildRecorderScene"/> writes the reference-time scene.</summary>
        public const string RecorderScenePath = SceneDirectory + "/BaselineRecorder.unity";

        /// <summary>Circuits per pool. Enough that the policy is learning the section, not the layouts.</summary>
        const int k_PoolSize = 100;

        /// <summary>
        /// Training areas per drill scene. Every one of them collects experience into the same
        /// policy each physics tick, so this is the parallelism a run gets inside one Editor
        /// session — matched to what the general training scene is set up with.
        /// </summary>
        const int k_Environments = 8;

        /// <summary>The car that goes on the start line in every area.</summary>
        const string k_CarPrefabPath = "Assets/RacingBotCup/Prefabs/MyCar.prefab";

        /// <summary>
        /// The agent parented under it. This is the policy being trained, so it has to be the same
        /// prefab the general training scene runs — drilling one entry and scoring another would
        /// train a policy nobody submits.
        /// </summary>
        const string k_AgentPrefabPath = "Assets/RacingBotCup/Prefabs/RacerAgentJH.prefab";

        /// <summary>
        /// First seed every scan looks at. Well clear of the ten evaluation seeds (3009–3307), so a
        /// drill cannot quietly turn into practice on the scored circuits.
        /// </summary>
        const int k_FirstSeed = 10000;

        /// <summary>
        /// Ceiling on how many seeds a scan will look at. The rarest filters here match about a
        /// quarter of seeds, so 100 hits need roughly 400 — this leaves an order of magnitude of
        /// headroom and still bounds the work if a filter turns out to match almost nothing.
        /// </summary>
        const int k_MaxScanned = 4000;

        /// <summary>
        /// One drill: which sections qualify a circuit, and what to call the scene and pool it
        /// produces. A circuit qualifies if it contains any one of <see cref="Sections"/>, which is
        /// what lets the mixed-hazard drill be an ordinary entry rather than a special case.
        /// </summary>
        public readonly struct Drill
        {
            public readonly string Id;
            public readonly string Purpose;
            public readonly TrackSectionType[] Sections;

            public Drill(string id, string purpose, params TrackSectionType[] sections)
            {
                Id = id;
                Purpose = purpose;
                Sections = sections;
            }

            public string ScenePath => $"{SceneDirectory}/{Id}.unity";

            public string SeedPoolPath => $"{SeedDirectory}/{ToSlug(Id)}_seeds.asset";
        }

        // ---- The drills ----------------------------------------------------
        // Hazard variants first: each is rolled at 25% per seed, so these are the three that random
        // practice leaves a policy least prepared for.

        static readonly Drill k_ObstacleStraight = new Drill(
            "ObstacleStraight",
            "crates, logs and containers scattered across a straight, with a passable gap that is " +
            "never in the same place twice",
            TrackSectionType.ObstacleStraight);

        static readonly Drill k_SharpHairpin = new Drill(
            "SharpHairpin",
            "the slowest corner on the circuit, pulled tighter than an ordinary hairpin and fenced " +
            "off on the inside by concrete barriers",
            TrackSectionType.SharpHairpin);

        static readonly Drill k_RampCorner = new Drill(
            "RampCorner",
            "a corner whose only racing line goes over a ramp, where the time is in controlling the " +
            "landing rather than in the corner itself",
            TrackSectionType.RampCorner);

        static readonly Drill k_MixedHazards = new Drill(
            "MixedHazards",
            "at least one hazard section of any kind, for putting the three individual drills back " +
            "together before returning to general practice",
            TrackSectionType.ObstacleStraight,
            TrackSectionType.SharpHairpin,
            TrackSectionType.RampCorner);

        // Common but not guaranteed, and both are places a policy loses time quietly rather than
        // spectacularly — worth drilling even though random practice does meet them.

        static readonly Drill k_Chicane = new Drill(
            "Chicane",
            "a short left-right flick at the end of a heavy braking zone, where going in too fast " +
            "puts the car in the opposite run-off",
            TrackSectionType.Chicane);

        static readonly Drill k_Esses = new Drill(
            "Esses",
            "a flowing alternating sequence where carrying momentum through the whole thing is the " +
            "entire game",
            TrackSectionType.Esses);

        static readonly Drill[] k_All =
        {
            k_ObstacleStraight,
            k_SharpHairpin,
            k_RampCorner,
            k_MixedHazards,
            k_Chicane,
            k_Esses,
        };

        // ---- Menu ----------------------------------------------------------

        [MenuItem("RacingBotCup/Drill Scenes/Obstacle Straight", priority = 10)]
        public static void BuildObstacleStraight() => BuildWithDialog(k_ObstacleStraight);

        [MenuItem("RacingBotCup/Drill Scenes/Sharp Hairpin", priority = 11)]
        public static void BuildSharpHairpin() => BuildWithDialog(k_SharpHairpin);

        [MenuItem("RacingBotCup/Drill Scenes/Ramp Corner", priority = 12)]
        public static void BuildRampCorner() => BuildWithDialog(k_RampCorner);

        [MenuItem("RacingBotCup/Drill Scenes/Mixed Hazards", priority = 13)]
        public static void BuildMixedHazards() => BuildWithDialog(k_MixedHazards);

        [MenuItem("RacingBotCup/Drill Scenes/Chicane", priority = 14)]
        public static void BuildChicane() => BuildWithDialog(k_Chicane);

        [MenuItem("RacingBotCup/Drill Scenes/Esses", priority = 15)]
        public static void BuildEsses() => BuildWithDialog(k_Esses);

        /// <summary>Builds every drill in one pass — a few seconds of scanning each.</summary>
        [MenuItem("RacingBotCup/Drill Scenes/Build All", priority = 26)]
        public static void BuildAll()
        {
            var report = new StringBuilder();

            foreach (var drill in k_All)
            {
                var pool = BuildSeedPool(drill);
                if (pool == null)
                {
                    report.AppendLine($"{drill.Id}: no qualifying seeds, skipped");
                    continue;
                }

                BuildScene(drill, pool);
                report.AppendLine($"{drill.Id}: {pool.Count} circuits");
            }

            EditorUtility.DisplayDialog(
                "RacingBot Cup",
                $"Drill scenes in {SceneDirectory}\n\n{report}",
                "OK");
        }

        /// <summary>
        /// Builds the scene that measures every drill pool's reference lap times, and opens it.
        ///
        /// A separate scene rather than a menu item that just runs, because the measurement needs
        /// play mode: it drives a real car around a real circuit with real physics, a hundred times
        /// per pool, and none of that exists in edit mode. Press Play and it records; it writes the
        /// pools and leaves play mode on its own.
        /// </summary>
        [MenuItem("RacingBotCup/Drill Scenes/Record Reference Times", priority = 27)]
        public static void BuildRecorderScene()
        {
            var pools = new List<SeedPool>();
            var missing = new List<string>();

            foreach (var drill in k_All)
            {
                var pool = AssetDatabase.LoadAssetAtPath<SeedPool>(drill.SeedPoolPath);
                if (pool != null)
                {
                    pools.Add(pool);
                }
                else
                {
                    missing.Add(drill.Id);
                }
            }

            if (pools.Count == 0)
            {
                EditorUtility.DisplayDialog(
                    "RacingBot Cup",
                    "No drill seed pools exist yet. Run Build All first — there is nothing to time " +
                    "until the pools have been chosen.",
                    "OK");
                return;
            }

            var car = AssetDatabase.LoadAssetAtPath<GameObject>(k_CarPrefabPath);
            var agent = AssetDatabase.LoadAssetAtPath<GameObject>(k_AgentPrefabPath);
            if (car == null || agent == null)
            {
                Debug.LogError($"[RacingBotCup] The recorder scene needs {k_CarPrefabPath} and {k_AgentPrefabPath}.");
                return;
            }

            Directory.CreateDirectory(SceneDirectory);

            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            SceneBootstrap.AddEnvironment();

            var recorderObject = new GameObject("BaselineTimeRecorder");
            var recorder = recorderObject.AddComponent<BaselineTimeRecorder>();

            var serialized = new SerializedObject(recorder);
            serialized.FindProperty("m_CarPrefab").objectReferenceValue = car;
            serialized.FindProperty("m_AgentPrefab").objectReferenceValue = agent;

            // The model is left unassigned — which checkpoint to trust as the reference is a call
            // only the caller can make, and Record() refuses to run without one rather than silently
            // falling back to something that could quietly become the wrong target.
            serialized.FindProperty("m_RunOnStart").boolValue = false;

            SceneBootstrap.AssignMaterials(serialized);
            SceneBootstrap.AssignProps(serialized);

            var poolList = serialized.FindProperty("m_Pools");
            poolList.arraySize = pools.Count;
            for (var i = 0; i < pools.Count; i++)
            {
                poolList.GetArrayElementAtIndex(i).objectReferenceValue = pools[i];
            }

            serialized.ApplyModifiedPropertiesWithoutUndo();

            EditorSceneManager.SaveScene(scene, RecorderScenePath);
            AssetDatabase.Refresh();

            var seeds = 0;
            foreach (var pool in pools)
            {
                seeds += pool.Count;
            }

            var note = missing.Count > 0
                ? $"\n\nNot built yet, so not measured: {string.Join(", ", missing)}."
                : "";

            EditorUtility.DisplayDialog(
                "RacingBot Cup",
                $"{RecorderScenePath} is open.\n\nSelect BaselineTimeRecorder and assign a trained " +
                ".onnx model as its reference — a fixed checkpoint, not the policy currently " +
                $"training. Then press Play to time {seeds} circuits across {pools.Count} pools " +
                "with it. It writes the times into the pools and leaves play mode when it is " +
                $"done.{note}",
                "OK");
        }

        // ---- Building ------------------------------------------------------

        static void BuildWithDialog(Drill drill)
        {
            var pool = BuildSeedPool(drill);
            if (pool == null)
            {
                EditorUtility.DisplayDialog(
                    "RacingBot Cup",
                    $"Found no circuits containing {Describe(drill)} in {k_MaxScanned} seeds. " +
                    "Nothing was written.",
                    "OK");
                return;
            }

            BuildScene(drill, pool);

            EditorUtility.DisplayDialog(
                "RacingBot Cup",
                $"Created:\n{drill.ScenePath}\n{drill.SeedPoolPath}\n\n" +
                $"{pool.Count} circuits, every one containing {Describe(drill)}.",
                "OK");
        }

        /// <summary>Scans for qualifying seeds and writes them to the drill's pool asset.</summary>
        public static SeedPool BuildSeedPool(Drill drill)
        {
            var seeds = SeedPoolBuilder.FindSeeds(
                drill.Sections, k_PoolSize, k_FirstSeed, k_MaxScanned, Describe(drill));

            if (seeds.Count == 0)
            {
                return null;
            }

            if (seeds.Count < k_PoolSize)
            {
                Debug.LogWarning(
                    $"[RacingBotCup] Wanted {k_PoolSize} circuits containing {Describe(drill)} but " +
                    $"found {seeds.Count} within {k_MaxScanned} seeds. The pool is still usable, " +
                    "just smaller.");
            }

            return SeedPoolBuilder.WriteAsset(
                drill.SeedPoolPath,
                seeds,
                $"{seeds.Count} practice circuits containing {Describe(drill)} — {drill.Purpose}. " +
                $"Found by walking seeds from {k_FirstSeed} with hazard sections enabled. " +
                "Regenerate from RacingBotCup > Drill Scenes.");
        }

        /// <summary>Rebuilds a drill's scene around an existing pool.</summary>
        public static void BuildScene(Drill drill, SeedPool pool)
        {
            Directory.CreateDirectory(SceneDirectory);

            if (PrefabBaker.LoadCarPrefab() == null)
            {
                PrefabBaker.RebuildPrefabs();
            }

            var options = BuildOptions();
            if (options == null)
            {
                return;
            }

            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            SceneBootstrap.AddEnvironment();

            for (var i = 0; i < k_Environments; i++)
            {
                // Opening on the first seeds of the pool rather than on one shared circuit, so the
                // scene shows a different example of the drill in every area before Play is pressed.
                SceneBootstrap.BuildTrainingEnvironment(i, pool.SeedAt(i), pool, options);
            }

            Verify(drill, pool);

            EditorSceneManager.SaveScene(scene, drill.ScenePath);
            AssetDatabase.Refresh();
        }

        /// <summary>
        /// Checks the scene actually came out drivable before it is saved.
        ///
        /// A drill scene fails quietly in two ways that both look fine in the hierarchy: an area
        /// whose car has no agent under it collects no experience at all, and an area whose seed
        /// pool did not get assigned falls back to the whole practice band — so the scene builds,
        /// opens, runs, and trains on the wrong circuits without ever saying so. Both have been seen
        /// in a generated scene, so they are worth a few lines to catch here rather than after a
        /// training run that went nowhere.
        /// </summary>
        static void Verify(Drill drill, SeedPool pool)
        {
            var problems = new List<string>();
            var arenas = UnityEngine.Object.FindObjectsByType<TrainingArena>(FindObjectsSortMode.None);

            // Not a problem the scene can fix, and not fatal — the arena falls back to a pace-based
            // estimate — but a drill whose reward is paid against an estimate rather than against a
            // time the baseline bot actually set is worth knowing about before the run, not after.
            if (pool.MeasuredCount == 0)
            {
                Debug.LogWarning(
                    $"[RacingBotCup] {drill.Id} has no reference lap times yet, so every episode " +
                    "will be scored against an estimate. Run RacingBotCup > Drill Scenes > " +
                    "Record Reference Times.");
            }

            if (arenas.Length != k_Environments)
            {
                problems.Add($"{arenas.Length} training areas, expected {k_Environments}");
            }

            foreach (var arena in arenas)
            {
                var where = arena.transform.parent != null ? arena.transform.parent.name : arena.name;
                var serialized = new SerializedObject(arena);

                if (serialized.FindProperty("m_SeedPool").objectReferenceValue != pool)
                {
                    problems.Add($"{where}: seed pool is not assigned, so it would train on random circuits");
                }

                if (serialized.FindProperty("m_Car").objectReferenceValue is not CarController car)
                {
                    problems.Add($"{where}: no car");
                }
                else if (car.GetComponentInChildren<RacerAgent>() == null)
                {
                    problems.Add($"{where}: no agent under {car.name}");
                }
            }

            if (problems.Count > 0)
            {
                Debug.LogError(
                    $"[RacingBotCup] {drill.Id} did not build cleanly:{Environment.NewLine}  " +
                    string.Join(Environment.NewLine + "  ", problems));
            }
        }

        /// <summary>
        /// The car, agent and area count a drill scene is built with — props come from the shared
        /// catalogue in <see cref="SceneBootstrap"/>, so a drill circuit carries exactly the
        /// obstacles every other scene does. Returns null after logging if a prefab is missing,
        /// rather than quietly building a scene with no car in it.
        /// </summary>
        static SceneBootstrap.EnvironmentOptions BuildOptions()
        {
            var car = AssetDatabase.LoadAssetAtPath<GameObject>(k_CarPrefabPath);
            var agent = AssetDatabase.LoadAssetAtPath<GameObject>(k_AgentPrefabPath);

            if (car == null || agent == null)
            {
                Debug.LogError(
                    $"[RacingBotCup] Drill scenes need {k_CarPrefabPath} and {k_AgentPrefabPath}. " +
                    "Point k_CarPrefabPath / k_AgentPrefabPath at the prefabs you train with.");
                return null;
            }

            return new SceneBootstrap.EnvironmentOptions
            {
                CarPrefab = car,
                AgentPrefab = agent,
                Count = k_Environments,
            };
        }

        /// <summary>"a RampCorner", or "an ObstacleStraight, SharpHairpin or RampCorner".</summary>
        static string Describe(Drill drill)
        {
            var names = new List<string>(drill.Sections.Length);
            foreach (var section in drill.Sections)
            {
                names.Add(section.ToString());
            }

            var article = names[0][0] == 'O' || names[0][0] == 'E' ? "an" : "a";
            if (names.Count == 1)
            {
                return $"{article} {names[0]}";
            }

            var last = names[names.Count - 1];
            names.RemoveAt(names.Count - 1);
            return $"{article} {string.Join(", ", names)} or {last}";
        }

        /// <summary>"ObstacleStraight" becomes "obstacle_straight", for asset filenames.</summary>
        static string ToSlug(string id)
        {
            var builder = new StringBuilder(id.Length + 4);
            foreach (var character in id)
            {
                if (char.IsUpper(character) && builder.Length > 0)
                {
                    builder.Append('_');
                }

                builder.Append(char.ToLowerInvariant(character));
            }

            return builder.ToString();
        }
    }
}

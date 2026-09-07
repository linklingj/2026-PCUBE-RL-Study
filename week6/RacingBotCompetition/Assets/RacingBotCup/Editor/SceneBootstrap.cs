using System;
using System.Collections.Generic;
using System.IO;
using RacingBotCup.Agent;
using RacingBotCup.Eval;
using RacingBotCup.Track;
using RacingBotCup.UI;
using RacingBotCup.Vehicle;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace RacingBotCup.EditorTools
{
    /// <summary>
    /// Builds the training and evaluation scenes from code.
    ///
    /// Both scenes come out ready to use: the training scene has a circuit and a car already on the
    /// start line, and the evaluation scene has one component with everything wired but the two
    /// fields a competitor fills in. Generating them from a menu means the setup is readable,
    /// diffable, and one click away from being restored.
    /// </summary>
    public static class SceneBootstrap
    {
        public const string SceneDirectory = "Assets/RacingBotCup/Scenes";
        public const string TrainingScenePath = SceneDirectory + "/Training.unity";
        public const string EvaluationScenePath = SceneDirectory + "/Evaluation.unity";

        const string k_ConfigDirectory = "Assets/RacingBotCup/Config";
        const string k_SeedSetPath = k_ConfigDirectory + "/eval_seeds.json";
        const string k_SubmissionConfigPath = k_ConfigDirectory + "/SubmissionConfig.asset";

        const string k_MaterialFolder = "Assets/PolygonStreetRacer/Materials/Roads/";

        // Picked by sampling the textures: Mat_03 is the darkest asphalt. The run-off takes its sand
        // colour from the swatch strip, which is the same in all of them. Grass is not in the pack,
        // so it is generated — see GeneratedMaterials.
        const string k_RoadMaterialPath = k_MaterialFolder + "PolygonStreetRacer_Road_Mat_03.mat";
        const string k_RunoffMaterialPath = k_MaterialFolder + "PolygonStreetRacer_Road_Mat_01.mat";

        const string k_PropFolder = "Assets/PolygonStreetRacer/Prefabs/Props/";

        // The prop catalogue below is the one every scene is built from, and the reason it is a
        // constant rather than something set per scene is that these prefabs are obstacles: their
        // colliders decide how wide the gap through an ObstacleStraight is and what a SharpHairpin
        // apex is fenced with. A scene whose catalogue drifted from this list is a scene whose lap
        // times cannot be compared with any other. RacingBotCup > Repair Scene Props pushes this
        // list back into every scene that already exists; new scenes get it by construction.

        static readonly string[] k_BarrierPrefabPaths =
        {
            k_PropFolder + "SM_Prop_Barrier_Concrete_02.prefab",
            k_PropFolder + "SM_Prop_Barrier_Concrete_02_Striped_01.prefab",
            k_PropFolder + "SM_Prop_Barrier_Concrete_03.prefab",
        };

        static readonly string[] k_CratePrefabPaths =
        {
            k_PropFolder + "SM_Prop_Crate_01.prefab",
            k_PropFolder + "SM_Prop_Crate_02.prefab",
        };

        static readonly string[] k_LogPrefabPaths =
        {
            k_PropFolder + "SM_Prop_Log_Single_01.prefab",
            k_PropFolder + "SM_Prop_Log_Single_02.prefab",
            k_PropFolder + "SM_Prop_Log_Single_05.prefab",
        };

        static readonly string[] k_ContainerPrefabPaths =
        {
            k_PropFolder + "SM_Prop_Container_Small_01.prefab",
            k_PropFolder + "SM_Prop_Container_Small_Doors_01.prefab",
        };

        const string k_RampPrefabPath = k_PropFolder + "SM_Prop_Ramp_Mesh_01.prefab";

        /// <summary>Circuit the training scene opens on. Any practice seed does.</summary>
        const int k_DefaultTrainingSeed = 4242;

        /// <summary>
        /// How many training areas run side by side. ML-Agents batches experience from every agent
        /// that shares a behaviour name, so replicating the whole environment — track, car and
        /// arena, not just the agent — is what actually parallelises training inside a single
        /// Editor session: four cars collecting steps every physics tick instead of one.
        /// </summary>
        const int k_ParallelEnvironments = 4;

        /// <summary>Exposed so a scene built elsewhere lays out the same number of areas.</summary>
        internal static int ParallelEnvironments => k_ParallelEnvironments;

        /// <summary>
        /// What one training scene may differ from another in. Every field left null or zero falls
        /// back to what the general training scene uses, so a caller only states what it changes —
        /// <see cref="DrillSceneBootstrap"/> asks for a different car and a different number of
        /// areas, and inherits everything else.
        /// </summary>
        internal sealed class EnvironmentOptions
        {
            /// <summary>Car prefab to drop on the start line. Null uses the baked one.</summary>
            public GameObject CarPrefab;

            /// <summary>Agent prefab parented under the car. Null uses the baked one.</summary>
            public GameObject AgentPrefab;

            /// <summary>How many areas the scene will hold. Only used to lay out the grid, so it has
            /// to match the number of times the caller builds one. Zero uses the default.</summary>
            public int Count;

            public GameObject ResolvedCarPrefab => CarPrefab != null ? CarPrefab : PrefabBaker.LoadCarPrefab();

            public GameObject ResolvedAgentPrefab => AgentPrefab != null ? AgentPrefab : PrefabBaker.LoadAgentPrefab();

            public int ResolvedCount => Count > 0 ? Count : k_ParallelEnvironments;
        }

        static readonly EnvironmentOptions k_DefaultEnvironment = new EnvironmentOptions();

        /// <summary>
        /// Distance between environments. Matches the spacing evaluation uses between its circuits —
        /// comfortably wider than the largest circuit a random seed can produce, so two training
        /// areas never overlap even as each one randomises independently every episode.
        /// </summary>
        const float k_EnvironmentSpacing = 1600f;

        [MenuItem("RacingBotCup/Build Scenes", priority = 0)]
        public static void BuildScenesFromMenu()
        {
            BuildScenes();
            EditorUtility.DisplayDialog(
                "RacingBot Cup",
                $"Created:\n{TrainingScenePath}\n{EvaluationScenePath}",
                "OK");
        }

        /// <summary>
        /// Reassigns materials and the prop catalogue in every scene that already exists, without
        /// regenerating them.
        ///
        /// This is the counterpart to <see cref="BuildScenes"/> for the case where a scene is worth
        /// keeping. A scene holds far more than its props — seeds, how many areas it has, which
        /// agent prefab is in them, whatever was tuned by hand — and rebuilding it to correct one
        /// list throws all of that away. Reaching into each <see cref="TrackInstance"/> and
        /// <see cref="RaceEvaluator"/> instead means the catalogue can be corrected in one place and
        /// pushed everywhere, which is the only way "every scene uses the same obstacles" stays true
        /// of scenes nobody wants to rebuild.
        ///
        /// Circuits are rebuilt afterwards because their obstacle geometry is baked into the scene:
        /// changing the catalogue without that leaves the old props sitting in the scene and only
        /// the next reroll picking up the new ones.
        /// </summary>
        [MenuItem("RacingBotCup/Repair Scene Props", priority = 41)]
        public static void RepairSceneProps()
        {
            var repaired = new List<string>();

            foreach (var path in ExistingScenePaths())
            {
                var scene = EditorSceneManager.OpenScene(path, OpenSceneMode.Single);
                var circuits = 0;

                foreach (var track in UnityEngine.Object.FindObjectsByType<TrackInstance>(FindObjectsSortMode.None))
                {
                    var serialized = new SerializedObject(track);
                    AssignMaterials(serialized);
                    AssignProps(serialized);
                    serialized.ApplyModifiedPropertiesWithoutUndo();

                    // The catalogue only reaches the scene's geometry through a rebuild.
                    track.Rebuild();
                    circuits++;
                }

                var evaluators = UnityEngine.Object.FindObjectsByType<RaceEvaluator>(FindObjectsSortMode.None);
                foreach (var evaluator in evaluators)
                {
                    var serialized = new SerializedObject(evaluator);
                    AssignMaterials(serialized);
                    AssignProps(serialized);
                    serialized.ApplyModifiedPropertiesWithoutUndo();
                }

                EditorSceneManager.MarkSceneDirty(scene);
                EditorSceneManager.SaveScene(scene);

                repaired.Add($"{Path.GetFileName(path)}: {circuits} circuit(s)" +
                             (evaluators.Length > 0 ? ", evaluator" : ""));
            }

            AssetDatabase.SaveAssets();

            var newLine = Environment.NewLine;
            EditorUtility.DisplayDialog(
                "RacingBot Cup",
                repaired.Count == 0
                    ? "No scenes found to repair."
                    : "Props reassigned from the catalogue in:" + newLine + newLine
                      + string.Join(newLine, repaired),
                "OK");
        }

        /// <summary>Every scene the project owns: the two built ones, plus each drill scene.</summary>
        static IEnumerable<string> ExistingScenePaths()
        {
            if (File.Exists(TrainingScenePath))
            {
                yield return TrainingScenePath;
            }

            if (File.Exists(EvaluationScenePath))
            {
                yield return EvaluationScenePath;
            }

            if (!Directory.Exists(DrillSceneBootstrap.SceneDirectory))
            {
                yield break;
            }

            var drills = Directory.GetFiles(DrillSceneBootstrap.SceneDirectory, "*.unity");
            Array.Sort(drills, string.CompareOrdinal);
            foreach (var drill in drills)
            {
                yield return drill.Replace(Path.DirectorySeparatorChar, '/');
            }
        }

        /// <summary>Rebuilds both scenes. Separate from the menu entry so automation can call it
        /// without a modal dialog blocking the editor.</summary>
        public static void BuildScenes()
        {
            Directory.CreateDirectory(SceneDirectory);
            Directory.CreateDirectory(k_ConfigDirectory);

            if (PrefabBaker.LoadCarPrefab() == null)
            {
                PrefabBaker.RebuildPrefabs();
            }

            EnsureSubmissionConfig();
            BuildTrainingScene();
            BuildEvaluationScene();

            AssetDatabase.Refresh();
        }

        static void BuildTrainingScene()
        {
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            AddEnvironment();

            for (var i = 0; i < k_ParallelEnvironments; i++)
            {
                BuildTrainingEnvironment(i);
            }

            EditorSceneManager.SaveScene(scene, TrainingScenePath);
        }

        static void BuildTrainingEnvironment(int index)
        {
            BuildTrainingEnvironment(index, k_DefaultTrainingSeed + index, null, null);
        }

        /// <summary>
        /// One self-contained training area: its own circuit, car, agent and arena, sitting far
        /// enough from the others that none of their geometry overlaps. Each starts on its own seed
        /// purely so the scene reads as four distinct environments the moment it opens — every arena
        /// still rerolls its own track independently once training starts.
        /// </summary>
        /// <param name="seedPool">
        /// Restricts what the arena may reroll to. Null lets it draw from the whole practice band,
        /// which is what the general training scene wants; a pool is how
        /// <see cref="DrillSceneBootstrap"/> keeps every episode on a circuit that contains the
        /// section its scene drills.
        /// </param>
        internal static void BuildTrainingEnvironment(
            int index, int seed, SeedPool seedPool, EnvironmentOptions options)
        {
            options ??= k_DefaultEnvironment;

            var root = new GameObject($"Environment_{index}");
            root.transform.position = GridPosition(index, options.ResolvedCount);

            var track = CreateTrack(seed, root.transform, options);
            var car = CreateCar(track, root.transform, options);

            var arenaObject = new GameObject("TrainingArena");
            arenaObject.transform.SetParent(root.transform, false);
            var component = arenaObject.AddComponent<TrainingArena>();

            var serialized = new SerializedObject(component);
            serialized.FindProperty("m_Track").objectReferenceValue = track;
            serialized.FindProperty("m_Car").objectReferenceValue = car;
            serialized.FindProperty("m_SeedPool").objectReferenceValue = seedPool;
            serialized.ApplyModifiedPropertiesWithoutUndo();
        }

        static Vector3 GridPosition(int index, int count)
        {
            var columns = Mathf.Max(1, Mathf.CeilToInt(Mathf.Sqrt(count)));
            return new Vector3(
                index % columns * k_EnvironmentSpacing,
                0f,
                index / columns * k_EnvironmentSpacing);
        }

        static void BuildEvaluationScene()
        {
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            AddEnvironment();

            var host = new GameObject("RaceEvaluator");
            var evaluator = host.AddComponent<RaceEvaluator>();

            var serialized = new SerializedObject(evaluator);
            serialized.FindProperty("m_CarPrefab").objectReferenceValue = PrefabBaker.LoadCarPrefab();
            serialized.FindProperty("m_AgentPrefab").objectReferenceValue = PrefabBaker.LoadAgentPrefab();
            serialized.FindProperty("m_SeedSetAsset").objectReferenceValue = LoadAsset<TextAsset>(k_SeedSetPath);
            serialized.FindProperty("m_SubmissionConfig").objectReferenceValue =
                LoadAsset<SubmissionConfig>(k_SubmissionConfigPath);
            serialized.FindProperty("m_RunOnStart").boolValue = true;
            AssignMaterials(serialized);
            AssignProps(serialized);
            serialized.ApplyModifiedPropertiesWithoutUndo();

            EditorSceneManager.SaveScene(scene, EvaluationScenePath);
        }

        /// <summary>Creates a circuit — parented, so a training area can be offset as one unit — and
        /// bakes its geometry into the scene.</summary>
        static TrackInstance CreateTrack(int seed, Transform parent, EnvironmentOptions options)
        {
            var trackObject = new GameObject("Circuit");
            trackObject.transform.SetParent(parent, false);
            var track = trackObject.AddComponent<TrackInstance>();

            var serialized = new SerializedObject(track);
            serialized.FindProperty("m_Seed").intValue = seed;
            AssignMaterials(serialized);
            AssignProps(serialized);
            serialized.ApplyModifiedPropertiesWithoutUndo();

            track.Rebuild();
            return track;
        }

        /// <summary>Drops the car on the start line with the agent already attached.</summary>
        static CarController CreateCar(TrackInstance track, Transform parent, EnvironmentOptions options)
        {
            var carPrefab = options.ResolvedCarPrefab;
            var agentPrefab = options.ResolvedAgentPrefab;

            if (carPrefab == null || agentPrefab == null)
            {
                Debug.LogError("[RacingBotCup] Prefabs are missing. Run RacingBotCup > Rebuild Prefabs.");
                return null;
            }

            var carObject = (GameObject)PrefabUtility.InstantiatePrefab(carPrefab);
            carObject.name = carPrefab.name;
            carObject.transform.SetParent(parent, false);

            var agentObject = (GameObject)PrefabUtility.InstantiatePrefab(agentPrefab);
            agentObject.name = "Agent";
            agentObject.transform.SetParent(carObject.transform, false);

            // The track's own model was already rebased to its (offset) transform, so this is a
            // world-space pose that lands the car correctly no matter which environment it is in.
            var pose = track.Model.GetStartPose(RaceRules.StartHeightOffset);
            carObject.transform.SetPositionAndRotation(pose.position, pose.rotation);

            return carObject.GetComponent<CarController>();
        }

        internal static void AddEnvironment()
        {
            var lightObject = new GameObject("Directional Light");
            var light = lightObject.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.1f;
            light.shadows = LightShadows.Soft;
            lightObject.transform.rotation = Quaternion.Euler(48f, 140f, 0f);

            var cameraObject = new GameObject("Main Camera") { tag = "MainCamera" };
            var camera = cameraObject.AddComponent<Camera>();
            camera.clearFlags = CameraClearFlags.Skybox;
            camera.farClipPlane = 3000f;
            cameraObject.transform.SetPositionAndRotation(
                new Vector3(0f, 40f, -60f),
                Quaternion.Euler(25f, 0f, 0f));

            // Following the car is the default: watching a lap from a fixed overhead camera means
            // watching a few pixels move.
            cameraObject.AddComponent<ChaseCamera>();
            cameraObject.AddComponent<RaceHud>();
        }

        static void EnsureSubmissionConfig()
        {
            if (AssetDatabase.LoadAssetAtPath<SubmissionConfig>(k_SubmissionConfigPath) != null)
            {
                return;
            }

            var config = ScriptableObject.CreateInstance<SubmissionConfig>();
            AssetDatabase.CreateAsset(config, k_SubmissionConfigPath);
        }

        internal static void AssignMaterials(SerializedObject serialized)
        {
            var materials = serialized.FindProperty("m_Materials");
            if (materials == null)
            {
                return;
            }

            materials.FindPropertyRelative("Road").objectReferenceValue =
                LoadAsset<Material>(k_RoadMaterialPath);
            materials.FindPropertyRelative("Runoff").objectReferenceValue =
                LoadAsset<Material>(k_RunoffMaterialPath);
            materials.FindPropertyRelative("Ground").objectReferenceValue =
                GeneratedMaterials.LoadOrCreateGrass();
        }

        /// <summary>Loads the circuit materials for tools that build a track outside a scene.</summary>
        public static TrackMaterials LoadMaterials()
        {
            return new TrackMaterials
            {
                Road = LoadAsset<Material>(k_RoadMaterialPath),
                Runoff = LoadAsset<Material>(k_RunoffMaterialPath),
                Ground = GeneratedMaterials.LoadOrCreateGrass(),
            };
        }

        internal static void AssignProps(SerializedObject serialized)
        {
            var props = serialized.FindProperty("m_Props");
            if (props == null)
            {
                return;
            }

            AssignPrefabArray(props.FindPropertyRelative("Barriers"), k_BarrierPrefabPaths);
            AssignPrefabArray(props.FindPropertyRelative("Crates"), k_CratePrefabPaths);
            AssignPrefabArray(props.FindPropertyRelative("Logs"), k_LogPrefabPaths);
            AssignPrefabArray(props.FindPropertyRelative("Containers"), k_ContainerPrefabPaths);
            props.FindPropertyRelative("Ramp").objectReferenceValue = LoadAsset<GameObject>(k_RampPrefabPath);
        }

        static void AssignPrefabArray(SerializedProperty arrayProperty, string[] paths)
        {
            if (arrayProperty == null)
            {
                return;
            }

            arrayProperty.arraySize = paths.Length;
            for (var i = 0; i < paths.Length; i++)
            {
                arrayProperty.GetArrayElementAtIndex(i).objectReferenceValue = LoadAsset<GameObject>(paths[i]);
            }
        }

        /// <summary>Loads the prop catalogue for tools that build a track outside a scene.</summary>
        public static TrackPropCatalogue LoadProps()
        {
            return new TrackPropCatalogue
            {
                Barriers = LoadAssets<GameObject>(k_BarrierPrefabPaths),
                Crates = LoadAssets<GameObject>(k_CratePrefabPaths),
                Logs = LoadAssets<GameObject>(k_LogPrefabPaths),
                Containers = LoadAssets<GameObject>(k_ContainerPrefabPaths),
                Ramp = LoadAsset<GameObject>(k_RampPrefabPath),
            };
        }

        static T[] LoadAssets<T>(string[] paths) where T : UnityEngine.Object
        {
            var results = new T[paths.Length];
            for (var i = 0; i < paths.Length; i++)
            {
                results[i] = LoadAsset<T>(paths[i]);
            }

            return results;
        }

        static T LoadAsset<T>(string path) where T : UnityEngine.Object
        {
            var asset = AssetDatabase.LoadAssetAtPath<T>(path);
            if (asset == null)
            {
                Debug.LogWarning($"[RacingBotCup] Missing asset at {path}; the scene will fall back to defaults.");
            }

            return asset;
        }
    }
}

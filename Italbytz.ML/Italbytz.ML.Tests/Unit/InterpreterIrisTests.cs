using System.Reflection;
using Italbytz.ML.Trainers;
using Italbytz.ML.Trainers.FastTree;
using JetBrains.Annotations;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers.FastTree;

namespace Italbytz.ML.Tests.Unit;

[TestClass]
[TestSubject(typeof(Interpreter))]
public class InterpreterIrisTests
{
    private readonly Interpreter _interpreter;
    private readonly ITransformer? _irisModel;

    public InterpreterIrisTests()
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        // Load the model
        var modelPath =
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models",
                "iris_20250617102903_100_Model.mlnet");
        _irisModel = mlContext.Model.Load(modelPath, out _);
        // Create the interpreter
        _interpreter = new Interpreter(_irisModel);
    }

    [TestMethod]
    public void TestExtractRegressionTrees()
    {
        var _timeStamp = DateTime.Now.ToString("yyyyMMddHHmmss");
        var filePrefix = "iris";
        var tmpDir = Path.GetTempPath();
        var interpretationsDir = Path.Combine(tmpDir, "interpretations");
        if (!Directory.Exists(interpretationsDir))
            Directory.CreateDirectory(interpretationsDir);
        var modelParameters =
            _interpreter.ExternalizedModelParameters;
        // ToDO: Currently hard coded for FastTreeBinaryModelParameters
        if (modelParameters is PublicOneVersusAllModelParameters pova)
            foreach (var submodel in pova.SubModelParameters)
                if (submodel is CalibratedModelParametersBase cmp)
                {
                    var subModelProperty =
                        typeof(CalibratedModelParametersBase).GetProperty(
                            "SubModel",
                            BindingFlags.DeclaredOnly | BindingFlags.Instance |
                            BindingFlags.NonPublic |
                            BindingFlags.Public | BindingFlags.Static);
                    if (subModelProperty != null)
                    {
                        var subModels = subModelProperty.GetValue(submodel);
                        if (subModels is FastTreeBinaryModelParameters
                            treeBinaryModelParameters)
                        {
                            var trees = treeBinaryModelParameters
                                .TrainedTreeEnsemble;
                            var index = 0;
                            foreach (var tree in trees.Trees)
                            {
                                var graphviz = tree.ToGraphviz();
                                var graphvizPath = Path.Combine(
                                    interpretationsDir,
                                    $"{filePrefix}_{_timeStamp}_Tree_{index}.dot");
                                File.WriteAllText(graphvizPath, graphviz);
                                var plantuml = tree.ToPlantUML();
                                var plantumlPath = Path.Combine(
                                    interpretationsDir,
                                    $"{filePrefix}_{_timeStamp}_Tree_{index}.pu");
                                File.WriteAllText(plantumlPath, plantuml);

                                index++;
                            }
                        }
                    }
                }
    }
}
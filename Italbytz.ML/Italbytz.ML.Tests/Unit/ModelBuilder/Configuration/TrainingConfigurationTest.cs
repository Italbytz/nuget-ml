using Italbytz.ML.ModelBuilder.Configuration;
using JetBrains.Annotations;

namespace Italbytz.ML.Tests.Unit.ModelBuilder.Configuration;

[TestClass]
[TestSubject(typeof(TrainingConfiguration))]
public class TrainingConfigurationTest
{
    [TestMethod]
    public void TestJsonSerialization()
    {
        ColumnPropertiesV5[] columnProperties =
        [
            new()
            {
                ColumnName = "c",
                ColumnDataFormat = ColumnDataKind.Boolean,
                ColumnPurpose = ColumnPurposeType.AnswerIndex,
                IsCategorical = false,
                Type = "T"
            }
        ];
        var dataSource = new TabularFileDataSourceV3
        {
            EscapeCharacter = '\\',
            ReadMultiLines = false,
            AllowQuoting = false,
            FilePath = "path",
            Delimiter = ",",
            DecimalMarker = '.',
            HasHeader = true,
            ColumnProperties = columnProperties
        };
        var environment = new LocalEnvironmentV1
        {
            Type = "LocalCPU",
            EnvironmentType = EnvironmentType.LocalCPU
        };
        var trainingOption = new ClassificationTrainingOptionV2
        {
            Subsampling = false,
            TrainingTime = 10,
            LabelColumn = "Label",
            AvailableTrainers = ["Trainer1", "Trainer2"],
            ValidationOption = new TrainValidationSplitOptionV0
            {
                SplitRatio = 0.1f
            }
        };
        var config = new TrainingConfiguration
        {
            Scenario = ScenarioType.Classification,
            DataSource = dataSource,
            Environment = environment,
            TrainingOption = trainingOption
        };
        var json = config.SerializeToJson();
        Assert.IsTrue(json.Contains("Trainer1"));
    }
}
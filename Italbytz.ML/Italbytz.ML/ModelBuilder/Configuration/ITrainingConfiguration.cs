using System.Text.Json.Serialization;

namespace Italbytz.ML.ModelBuilder.Configuration;

[JsonDerivedType(typeof(TrainingConfiguration))]
public interface ITrainingConfiguration
{
    public ScenarioType Scenario { get; set; }

    public IDataSource? DataSource { get; set; }

    public IEnvironment? Environment { get; set; }

    public ITrainingOption? TrainingOption { get; set; }

    public AutoMLType? AutoMLType { get; set; }

    public ITrainResult? TrainResult { get; set; }

    public string? TrainingConfigurationFolder { get; set; }

    public string SerializeToJson(bool writeIndented = false);
}
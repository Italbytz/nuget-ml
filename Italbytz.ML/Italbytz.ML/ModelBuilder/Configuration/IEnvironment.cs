using System.Text.Json.Serialization;

namespace Italbytz.ML.ModelBuilder.Configuration;

[JsonDerivedType(typeof(LocalEnvironmentV1))]
public interface IEnvironment
{
    EnvironmentType EnvironmentType { get; set; }
}
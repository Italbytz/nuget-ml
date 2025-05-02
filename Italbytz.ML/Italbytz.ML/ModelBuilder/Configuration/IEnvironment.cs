using System.Text.Json.Serialization;
using Italbytz.ML.ModelBuilder.Configuration;

namespace logicGP.Tests.Util.ML.ModelBuilder.Configuration;

[JsonDerivedType(typeof(LocalEnvironmentV1))]
public interface IEnvironment
{
    EnvironmentType EnvironmentType { get; set; }
}
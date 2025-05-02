using System.Text.Json.Serialization;
using Italbytz.ML.ModelBuilder.Configuration;

namespace logicGP.Tests.Util.ML.ModelBuilder.Configuration;

[JsonDerivedType(typeof(TabularFileDataSourceV3))]
public interface IDataSource
{
    DataSourceType DataSourceType { get; set; }
}
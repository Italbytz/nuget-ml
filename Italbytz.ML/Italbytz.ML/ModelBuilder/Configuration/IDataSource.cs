using System.Text.Json.Serialization;

namespace Italbytz.ML.ModelBuilder.Configuration;

[JsonDerivedType(typeof(TabularFileDataSourceV3))]
public interface IDataSource
{
    DataSourceType DataSourceType { get; set; }
}
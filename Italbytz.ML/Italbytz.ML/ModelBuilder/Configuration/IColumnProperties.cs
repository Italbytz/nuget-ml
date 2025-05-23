using System.Text.Json.Serialization;

namespace Italbytz.ML.ModelBuilder.Configuration;

[JsonDerivedType(typeof(ColumnPropertiesV5))]
public interface IColumnProperties
{
    string? ColumnName { get; set; }

    ColumnPurposeType ColumnPurpose { get; set; }

    ColumnDataKind ColumnDataFormat { get; set; }

    bool IsCategorical { get; set; }
}
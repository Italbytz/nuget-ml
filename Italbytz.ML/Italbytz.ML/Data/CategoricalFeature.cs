namespace Italbytz.ML.Data;

public class CategoricalFeature : IValueRangeFeature<string>
{
    public string PropertyName { get; set; }
    public string ColumnName { get; set; }
    public int ColumnIndex { get; set; }
    public List<string> ValueRange { get; set; }
}
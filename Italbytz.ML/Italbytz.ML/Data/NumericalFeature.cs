namespace Italbytz.ML.Data;

public class NumericalFeature : IValueRangeFeature<float>
{
    public List<float> ValueRange { get; set; }
    public string PropertyName { get; set; }
    public string ColumnName { get; set; }
    public int ColumnIndex { get; set; }
}
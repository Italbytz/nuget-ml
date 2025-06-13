namespace Italbytz.ML.Data;

public interface IFeature
{
    public string PropertyName { get; set; }
    public string ColumnName { get; set; }
    public int ColumnIndex { get; set; }
}
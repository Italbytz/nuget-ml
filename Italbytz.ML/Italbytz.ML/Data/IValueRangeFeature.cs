namespace Italbytz.ML.Data;

public interface IValueRangeFeature<ValueType> : IFeature
{
    List<ValueType> ValueRange { get; set; }
}
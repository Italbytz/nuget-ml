namespace Italbytz.ML.ModelBuilder.Configuration;

public interface ITabularDataSource : IDataSource
{
    IEnumerable<IColumnProperties> ColumnProperties { get; set; }
}
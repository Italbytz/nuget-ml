using JetBrains.Annotations;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Tests.Unit;

[TestClass]
[TestSubject(typeof(IDataViewExtensions))]
public class IDataViewExtensionsTest
{
    private readonly IDataView? _data;

    public IDataViewExtensionsTest()
    {
        var inMemoryCollection = new[]
        {
            new HousingData
            {
                Size = 700f,
                HistoricalPrices =
                [
                    100000f, 3000000f, 250000f
                ],
                CurrentPrice = 500000f
            },
            new HousingData
            {
                Size = 1000.4f,
                HistoricalPrices =
                [
                    600000.1f, 400000.2f, 650000f
                ],
                CurrentPrice = 700000.3f
            }
        };
        // Create MLContext
        var mlContext = new MLContext();
        //Load Data
        _data = mlContext.Data.LoadFromEnumerable(inMemoryCollection);
    }

    [TestMethod]
    public void TestSaveAsCsv()
    {
        // Arrange
        var mlContext = new MLContext();
        var filePath = Path.Combine(Path.GetTempPath(), "HousingData.csv");

        // Act
        _data?.SaveAsCsv(filePath);

        // Assert
        Assert.IsTrue(File.Exists(filePath));

        // Load the data back from the CSV file
        var loadedData = mlContext.Data.LoadFromTextFile<HousingData>(
            filePath, ',', true);

        var loadedDataView = loadedData.GetColumn<float>("Size").ToArray();
        Assert.AreEqual(2, loadedDataView.Length);

        // Clean up
        File.Delete(filePath);
    }

    [TestMethod]
    public void TestToDataTable()
    {
        // Arrange
        var dataTable = _data?.ToDataTable();

        // Act
        Assert.IsNotNull(dataTable);

        // Assert
        Assert.AreEqual(2, dataTable?.Rows.Count);
        Assert.AreEqual(3, dataTable?.Columns.Count);
        Assert.AreEqual("Size", dataTable?.Columns[0].ColumnName);
        Assert.AreEqual("HistoricalPrices", dataTable?.Columns[1].ColumnName);
        Assert.AreEqual("Label", dataTable?.Columns[2].ColumnName);
    }

    [TestMethod]
    public void TestWriteToCsv()
    {
        // Arrange
        var mlContext = new MLContext();
        var filePath = Path.Combine(Path.GetTempPath(), "HousingData.csv");

        // Act
        _data?.WriteToCsv(filePath);

        // Assert
        Assert.IsTrue(File.Exists(filePath));

        // Load the data back from the CSV file
        var loadedData = mlContext.Data.LoadFromTextFile<HousingData>(
            filePath, ',', true);

        var loadedDataView = loadedData.GetColumn<float>("Size").ToArray();
        Assert.AreEqual(2, loadedDataView.Length);

        // Clean up
        File.Delete(filePath);
    }
}

public class HousingData
{
    [LoadColumn(0)] public float Size { get; set; }

    [LoadColumn(1, 3)]
    [VectorType(3)]
    public float[] HistoricalPrices { get; set; }

    [LoadColumn(4)]
    [ColumnName("Label")]
    public float CurrentPrice { get; set; }
}
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
                Size = 700f, HistoricalPrices = [100000f, 3000000f, 250000f],
                CurrentPrice = 500000f
            },
            new HousingData
            {
                Size = 1000.4f,
                HistoricalPrices = [600000.1f, 400000.2f, 650000f],
                CurrentPrice = 700000.3f
            },
            new HousingData
            {
                Size = 850f, HistoricalPrices = [200000f, 250000f, 300000f],
                CurrentPrice = 275000f
            },
            new HousingData
            {
                Size = 1200f, HistoricalPrices = [400000f, 420000f, 450000f],
                CurrentPrice = 430000f
            },
            new HousingData
            {
                Size = 950f, HistoricalPrices = [320000f, 330000f, 340000f],
                CurrentPrice = 335000f
            },
            new HousingData
            {
                Size = 1100f, HistoricalPrices = [500000f, 510000f, 520000f],
                CurrentPrice = 515000f
            },
            new HousingData
            {
                Size = 780f, HistoricalPrices = [210000f, 215000f, 220000f],
                CurrentPrice = 218000f
            },
            new HousingData
            {
                Size = 1300f, HistoricalPrices = [600000f, 610000f, 620000f],
                CurrentPrice = 615000f
            },
            new HousingData
            {
                Size = 900f, HistoricalPrices = [290000f, 295000f, 300000f],
                CurrentPrice = 298000f
            },
            new HousingData
            {
                Size = 1050f, HistoricalPrices = [410000f, 420000f, 430000f],
                CurrentPrice = 425000f
            },
            new HousingData
            {
                Size = 1150f, HistoricalPrices = [530000f, 540000f, 550000f],
                CurrentPrice = 545000f
            },
            new HousingData
            {
                Size = 980f, HistoricalPrices = [350000f, 355000f, 360000f],
                CurrentPrice = 358000f
            },
            new HousingData
            {
                Size = 1020f, HistoricalPrices = [370000f, 375000f, 380000f],
                CurrentPrice = 378000f
            },
            new HousingData
            {
                Size = 1250f, HistoricalPrices = [650000f, 660000f, 670000f],
                CurrentPrice = 665000f
            },
            new HousingData
            {
                Size = 890f, HistoricalPrices = [275000f, 280000f, 285000f],
                CurrentPrice = 282000f
            },
            new HousingData
            {
                Size = 1400f, HistoricalPrices = [700000f, 710000f, 720000f],
                CurrentPrice = 715000f
            },
            new HousingData
            {
                Size = 1005f, HistoricalPrices = [360000f, 365000f, 370000f],
                CurrentPrice = 368000f
            },
            new HousingData
            {
                Size = 1080f, HistoricalPrices = [430000f, 440000f, 450000f],
                CurrentPrice = 445000f
            },
            new HousingData
            {
                Size = 950.5f, HistoricalPrices = [310000f, 315000f, 320000f],
                CurrentPrice = 318000f
            },
            new HousingData
            {
                Size = 1205f, HistoricalPrices = [480000f, 490000f, 500000f],
                CurrentPrice = 495000f
            }
        };
        // Create MLContext
        var mlContext = new MLContext();
        //Load Data
        _data = mlContext.Data.LoadFromEnumerable(inMemoryCollection);
    }

    [TestMethod]
    public void TestGenerateTrainValidateTestCsvsNoSeed()
    {
        var folder = Path.GetTempPath();
        var files = _data.GenerateTrainValidateTestCsvs(folder, "HousingData",
            validateFraction: 0.1, testFraction: 0.1);
        Assert.IsNotNull(files);
        Assert.AreEqual(1, files.Count());
        Assert.IsTrue(
            File.Exists(Path.Combine(folder, files.First().TrainFileName)));
        Assert.IsTrue(
            File.Exists(Path.Combine(folder, files.First().ValidateFileName)));
        Assert.IsTrue(
            File.Exists(Path.Combine(folder, files.First().TestFileName)));
    }

    [TestMethod]
    public void TestGenerateTrainValidateTestCsvs()
    {
        var folder = Path.GetTempPath();
        var files = _data.GenerateTrainValidateTestCsvs(folder, "HousingData",
            validateFraction: 0.1, testFraction: 0.1, seeds: [1, 2, 3]);
        Assert.IsNotNull(files);
        Assert.AreEqual(3, files.Count());
    }

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
        Assert.AreEqual(20, loadedDataView.Length);

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
        Assert.AreEqual(20, dataTable?.Rows.Count);
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
        Assert.AreEqual(20, loadedDataView.Length);

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
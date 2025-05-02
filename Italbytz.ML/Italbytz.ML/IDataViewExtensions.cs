using System.Collections.Immutable;
using System.Data;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML;

/// <summary>
///     Extensions for <see cref="IDataView" />.
/// </summary>
/// <remarks>
///     This class contains extension methods for the <see cref="IDataView" />
///     interface.
///     It provides methods to convert an <see cref="IDataView" /> to a
///     <see cref="DataTable" />, save it as CSV,
///     and retrieve unique column entries.
/// </remarks>
public static class IDataViewExtensions
{
    /// <summary>
    ///     Converts an <see cref="IDataView" /> to a <see cref="DataTable" />.
    /// </summary>
    /// <returns>
    ///     A <see cref="DataTable" /> representing the data in the
    ///     <see cref="IDataView" />.
    /// </returns>
    /// <remarks>
    ///     This method creates a new <see cref="DataTable" /> and populates it with
    ///     the data from the <see cref="IDataView" />.
    ///     The columns of the <see cref="DataTable" /> are created based on the schema
    ///     of the <see cref="IDataView" />.
    ///     Each row in the <see cref="DataTable" /> corresponds to a row in the
    ///     <see cref="IDataView" />.
    ///     The method uses the <see cref="IDataView.Preview" /> method to get a
    ///     preview of the data and its schema.
    /// </remarks>
    public static DataTable? ToDataTable(this IDataView? dataView)
    {
        DataTable? dt = null;
        if (dataView == null) return dt;
        dt = new DataTable();
        var preview = dataView.Preview();
        dt.Columns.AddRange(preview.Schema
            .Select(x => new DataColumn(x.Name)).ToArray());
        foreach (var row in preview.RowView)
        {
            var r = dt.NewRow();
            foreach (var col in row.Values) r[col.Key] = col.Value;
            dt.Rows.Add(r);
        }

        return dt;
    }

    /// <summary>
    ///     Saves an <see cref="IDataView" /> as a CSV file.
    /// </summary>
    /// <param name="filePath">The file path where the CSV file will be saved.</param>
    /// <remarks>
    ///     This method uses the <see cref="MLContext.Data.SaveAsText" /> method to
    ///     save the <see cref="IDataView" /> as a CSV file.
    ///     The CSV file will be created at the specified file path.
    ///     The method uses a comma (',') as the separator character and does not
    ///     include the schema in the output.
    /// </remarks>
    public static void SaveAsCsv(
        this IDataView dataView,
        string filePath
    )
    {
        using var dataStream = new FileStream(
            filePath,
            FileMode.Create, FileAccess.Write);
        new MLContext().Data.SaveAsText(dataView, dataStream, ',',
            schema: false);
    }

    /// <summary>
    ///     Writes an <see cref="IDataView" /> to a CSV file.
    /// </summary>
    /// <param name="filePath">The file path where the CSV file will be saved.</param>
    /// <remarks>
    ///     This method creates a CSV file at the specified file path.
    ///     The CSV file will contain the data from the <see cref="IDataView" />.
    ///     The first line of the CSV file will contain the column names.
    ///     Each subsequent line will contain the values of a row in the
    ///     <see cref="IDataView" />.
    ///     The method uses a comma (',') as the separator character.
    /// </remarks>
    public static void WriteToCsv(
        this IDataView dataView,
        string filePath
    )
    {
        var dt = dataView.ToDataTable();

        var sb = new StringBuilder();

        Debug.Assert(dt != null, nameof(dt) + " != null");
        var columnNames = dt.Columns.Cast<DataColumn>()
            .Select(column => column.ColumnName);
        sb.AppendLine(string.Join(",", columnNames));

        foreach (DataRow row in dt.Rows)
        {
            var fields =
                row.ItemArray.Select(InvariantCultureString);
            sb.AppendLine(string.Join(",", fields));
        }

        File.WriteAllText(filePath, sb.ToString());
    }

    private static string InvariantCultureString(object field)
    {
        var returnValue = field.GetType() switch
        {
            { } floatType when floatType == typeof(float) => ((float)field)
                .ToString(CultureInfo.InvariantCulture),
            { } doubleType when doubleType == typeof(double) => ((double)field)
                .ToString(CultureInfo.InvariantCulture),
            _ => field.ToString()
        };

        return returnValue;
    }

    /// <summary>
    ///     Gets the slot names of the features column in the <see cref="IDataView" />.
    /// </summary>
    /// <param name="columnName">
    ///     The name of the features column. Default is
    ///     "Features".
    /// </param>
    /// <returns>An array of slot names.</returns>
    /// <remarks>
    ///     This method retrieves the slot names from the features column of the
    ///     <see cref="IDataView" />.
    ///     The slot names are stored as annotations in the features column.
    ///     The method uses the <see cref="DataViewSchema.Column.Annotations" />
    ///     property to access the annotations.
    ///     If the features column does not exist or does not contain annotations, an
    ///     exception is thrown.
    ///     The slot names are returned as an array of
    ///     <see cref="ReadOnlyMemory{char}" />.
    /// </remarks>
    /// <exception cref="ArgumentException">
    ///     Thrown when the features column does not exist or does not contain
    ///     annotations.
    /// </exception>
    public static ImmutableArray<ReadOnlyMemory<char>> GetFeaturesSlotNames(
        this IDataView dataView,
        string columnName = "Features"
    )
    {
        var featuresColumn = dataView.Schema.GetColumnOrNull(columnName);
        if (featuresColumn == null)
            throw new ArgumentException(
                "The data view does not contain a column named 'Features'.");
        var featuresAnnotations = featuresColumn?.Annotations;
        if (featuresAnnotations == null)
            throw new ArgumentException(
                "The 'Features' column does not contain annotations.");
        VBuffer<ReadOnlyMemory<char>> slotNames = default;
        featuresAnnotations.GetValue("SlotNames", ref slotNames);
        return [..slotNames.GetValues()];
    }

    /// <summary>
    ///     Gets the unique entries in a specified column of an
    ///     <see cref="IDataView" />.
    /// </summary>
    /// <param name="columnName">The name of the column to retrieve the entries from.</param>
    /// <returns>An enumerable of unique entries in the specified column.</returns>
    /// <remarks>
    ///     This method retrieves the unique entries in a specified column of an
    ///     <see cref="IDataView" />.
    ///     The entries are stored as strings in the column.
    ///     The method uses the <see cref="GetColumnAsString" /> method to convert the
    ///     column data to strings.
    ///     The unique entries are returned as an enumerable of strings.
    /// </remarks>
    public static IEnumerable<string>? GetOrderedUniqueColumnEntries(
        this IDataView dataView,
        string columnName
    )
    {
        var labelColumn = dataView.Schema[columnName];
        var labelColumnData =
            (GetColumnAsString(dataView, labelColumn) ??
             throw new InvalidOperationException(
                 $"Column {columnName} can not be read as strings.")).ToList();
        return new HashSet<string>(
            labelColumnData).OrderBy(c => c);
    }

    /// <summary>
    ///     Gets the entries in a specified column of an <see cref="IDataView" /> as
    ///     strings.
    /// </summary>
    /// <param name="columnName">The name of the column to retrieve the entries from.</param>
    /// <returns>An enumerable of entries in the specified column as strings.</returns>
    /// <remarks>
    ///     This method retrieves the entries in a specified column of an
    ///     <see cref="IDataView" /> as strings.
    ///     The entries are converted to strings based on their type.
    ///     The method uses the <see cref="GetColumnAsString" /> method to convert the
    ///     column data to strings.
    ///     The entries are returned as an enumerable of strings.
    ///     The method handles different data types, including float, int, uint, char,
    ///     and string.
    ///     If the column type is not supported, it falls back to using the string
    ///     representation of the data.
    /// </remarks>
    public static IEnumerable<string>? GetColumnAsString(
        this IDataView dataView,
        string columnName
    )
    {
        var column = dataView.Schema[columnName];

        return GetColumnAsString(dataView, column);
    }

    /// <summary>
    ///     Gets the entries in a specified column of an <see cref="IDataView" /> as
    ///     strings.
    /// </summary>
    /// <param name="column">The column to retrieve the entries from.</param>
    /// <returns>An enumerable of entries in the specified column as strings.</returns>
    /// <remarks>
    ///     This method retrieves the entries in a specified column of an
    ///     <see cref="IDataView" /> as strings.
    ///     The entries are converted to strings based on their type.
    ///     The entries are returned as an enumerable of strings.
    ///     The method handles different data types, including float, int, uint, char,
    ///     and string.
    ///     If the column type is not supported, it falls back to using the string
    ///     representation of the data.
    /// </remarks>
    public static IEnumerable<string>? GetColumnAsString(
        this IDataView dataView, DataViewSchema.Column column)
    {
        var dataColumn = column.Type.RawType switch
        {
            { } floatType when floatType == typeof(float) => dataView
                .GetColumn<float>(column)
                .Select(entry => entry.ToString(CultureInfo.InvariantCulture)),
            { } intType when intType == typeof(int) => dataView
                .GetColumn<int>(column)
                .Select(entry => entry.ToString()),
            { } uintType when uintType == typeof(uint) => dataView
                .GetColumn<uint>(column)
                .Select(entry => entry.ToString()),
            { } charType when charType == typeof(char) => dataView
                .GetColumn<char>(column)
                .Select(entry => entry.ToString()),
            { } stringType when stringType == typeof(string) => dataView
                .GetColumn<string>(column),
            _ => dataView
                .GetColumn<string>(column)
        };
        return dataColumn;
    }

    
}
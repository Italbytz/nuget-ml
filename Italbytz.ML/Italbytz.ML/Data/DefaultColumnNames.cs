namespace Italbytz.ML;

/// <summary>
///     A set of string literals intended to be "canonical" names for column names
///     intended for particular purpose. Its primary purpose is intended to be used
///     in such a way as to encourage
///     uniformity, wherever it is judged where columns
///     with default names should be consumed.
/// </summary>
public static class DefaultColumnNames
{
    /// <summary>
    ///     The canonical column name for feature vectors.
    /// </summary>
    public const string Features = "Features";

    /// <summary>
    ///     The canonical column name for labels in supervised learning.
    /// </summary>
    public const string Label = "Label";

    /// <summary>
    ///     The canonical column name for grouping data points.
    /// </summary>
    public const string GroupId = "GroupId";

    /// <summary>
    ///     The canonical column name for the name of an entity.
    /// </summary>
    public const string Name = "Name";

    /// <summary>
    ///     The canonical column name for weights associated with data points.
    /// </summary>
    public const string Weight = "Weight";

    /// <summary>
    ///     The canonical column name for scores or ratings.
    /// </summary>
    public const string Score = "Score";

    /// <summary>
    ///     The canonical column name for probabilistic output.
    /// </summary>
    public const string Probability = "Probability";

    /// <summary>
    ///     The canonical column name for predicted labels in supervised learning.
    /// </summary>
    public const string PredictedLabel = "PredictedLabel";

    /// <summary>
    ///     The canonical column name for recommended items in a recommendation system.
    /// </summary>
    public const string RecommendedItems = "Recommended";

    /// <summary>
    ///     The canonical column name for users in a recommendation system.
    /// </summary>
    public const string User = "User";

    /// <summary>
    ///     The canonical column name for items in a recommendation system.
    /// </summary>
    public const string Item = "Item";

    /// <summary>
    ///     The canonical column name for date information.
    /// </summary>
    public const string Date = "Date";

    /// <summary>
    ///     The canonical column name for feature importance or contribution values.
    /// </summary>
    public const string FeatureContributions = "FeatureContributions";
}
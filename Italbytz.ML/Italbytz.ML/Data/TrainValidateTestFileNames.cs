namespace Italbytz.ML.Data;

/// <summary>
///     Represents file names for training, validation, and test datasets.
/// </summary>
public record TrainValidateTestFileNames
{
    /// <summary>
    ///     Gets or sets the file name for the training dataset.
    /// </summary>
    public required string TrainFileName { get; set; }


    /// <summary>
    ///     Gets or sets the file name for the validation dataset.
    /// </summary>
    public required string ValidateFileName { get; set; }

    /// <summary>
    ///     Gets or sets the file name for the training and validation dataset.
    /// </summary>
    public required string TrainValidateFileName { get; set; }

    /// <summary>
    ///     Gets or sets the file name for the test dataset.
    /// </summary>
    public required string TestFileName { get; set; }
}
using Microsoft.ML;

namespace Italbytz.ML;

/// <summary>
///     Interpreter class for handling ML.NET ITransformer models.
/// </summary>
/// <param name="model">The trained ML.NET model to be interpreted.</param>
public class Interpreter(
    ITransformer model)
{
    /// <summary>
    ///     Externalizes and returns the model parameters from the trained ML.NET
    ///     model.
    /// </summary>
    public ICanSaveModel ExternalizedModelParameters =>
        model.GetModelParameters();
}
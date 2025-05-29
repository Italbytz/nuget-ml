using Italbytz.ML.Trainers;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace Italbytz.ML;

/// <summary>
///     Extension methods for <see cref="ITransformer" /> to extract model
///     parameters.
/// </summary>
public static class ITransformerExtensions
{
    /// <summary>
    ///     Retrieves the model parameters from the given <see cref="ITransformer" />.
    /// </summary>
    /// <returns>The model parameters.</returns>
    public static ICanSaveModel GetModelParameters(
        this ITransformer transformer)
    {
        IPredictionTransformer<ICanSaveModel>? predictionTransformer = null;
        switch (transformer)
        {
            case IEnumerable<ITransformer> chain:
            {
                foreach (var chainItem in chain)
                    if (chainItem is IPredictionTransformer<ICanSaveModel>
                        predTransformer)
                    {
                        predictionTransformer = predTransformer;
                        break;
                    }

                break;
            }
            case IPredictionTransformer<ICanSaveModel>
                predTransformer:
                predictionTransformer = predTransformer;
                break;
        }

        var model = predictionTransformer.Model;
        if (model is OneVersusAllModelParameters oneVersusAllModelParameters)
            model = oneVersusAllModelParameters.ToPublic();
        return model;
    }
}
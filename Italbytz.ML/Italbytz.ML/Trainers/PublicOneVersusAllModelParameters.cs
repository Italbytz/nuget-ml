using System.Collections.Immutable;
using Microsoft.ML;

namespace Italbytz.ML.Trainers;

/// <summary>
///     Represents the model parameters for a One-Versus-All trainer.
///     Necessary for exposing the originally internal SubModelParameters
/// </summary>
public class
    PublicOneVersusAllModelParameters : ICanSaveModel
{
    /// <summary>
    ///     Gets or sets the parameters of the sub-models.
    /// </summary>
    public ImmutableArray<object> SubModelParameters { get; set; }


    /// <inheritdoc />
    public void Save(ModelSaveContext ctx)
    {
        throw new NotImplementedException();
    }
}
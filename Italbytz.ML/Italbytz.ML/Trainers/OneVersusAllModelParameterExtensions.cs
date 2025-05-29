using System.Collections.Immutable;
using System.Reflection;
using Microsoft.ML.Trainers;

namespace Italbytz.ML.Trainers;

/// <summary>
///     Provides extension methods for converting
///     <see cref="OneVersusAllModelParameters" />
///     to <see cref="PublicOneVersusAllModelParameters" />.
/// </summary>
public static class OneVersusAllModelParameterExtensions
{
    /// <summary>
    ///     Converts an instance of <see cref="OneVersusAllModelParameters" /> to a
    ///     <see cref="PublicOneVersusAllModelParameters" /> representation.
    /// </summary>
    /// <returns>A public representation of the model parameters.</returns>
    public static PublicOneVersusAllModelParameters ToPublic(
        this OneVersusAllModelParameters modelParameters)
    {
        var publicModelParameters =
            new PublicOneVersusAllModelParameters();
        var subModelParamsProp = modelParameters?.GetType()
            .GetProperty("SubModelParameters",
                BindingFlags.Instance | BindingFlags.NonPublic |
                BindingFlags.Public);
        if (subModelParamsProp != null)
        {
            if (subModelParamsProp.GetValue(modelParameters) is
                IEnumerable<object> subModelParams)
                publicModelParameters.SubModelParameters =
                    [..subModelParams];
        }
        else
        {
            publicModelParameters.SubModelParameters =
                ImmutableArray<object>.Empty;
        }

        return publicModelParameters;
    }
}
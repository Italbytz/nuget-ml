using System;
using System.Runtime.CompilerServices;
using Microsoft.ML;

namespace Italbytz.ML;

/// <summary>
///     Thread-safe wrapper for MLContext.
///     This helps in getting MLContext instances with a specific seed
///     in a thread-safe manner.
///     It uses ThreadStatic to ensure that each thread has its own instance of
///     MLContext.
///     This is useful when you want to use ML.NET in a multi-threaded environment
///     and need to ensure that the random number generation is consistent across
///     threads.
/// </summary>
public class ThreadSafeMLContext
{
    private static int? _seed;
    [ThreadStatic] private static MLContext? _tMLContext;

    /// <summary>
    ///     Gets or sets the seed for random number generation.
    /// </summary>
    public static int? Seed
    {
        get => _seed;
        set
        {
            _seed = value;
            _tMLContext =
                null; // Reset to ensure next access creates a new Random with the seed
        }
    }

    /// <summary>
    ///     Gets the thread-safe MLContext instance.
    /// </summary>
    /// <remarks>
    ///     This property is thread-safe and will create a new MLContext instance
    ///     for each thread that accesses it. The instance will be created with the
    ///     specified seed if it is set, otherwise a new instance will be created
    ///     without a seed.
    /// </remarks>

    public static MLContext LocalMLContext => _tMLContext ?? Create();


    [MethodImpl(MethodImplOptions.NoInlining)]
    private static MLContext Create()
    {
        return _tMLContext =
            _seed.HasValue ? new MLContext(_seed.Value) : new MLContext();
    }
}
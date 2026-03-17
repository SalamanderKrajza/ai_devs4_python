/** Check if error is from abort */
export function isAbortError(err) {
    return err instanceof Error && err.name === 'AbortError';
}
/** Throw if signal is aborted */
export function throwIfAborted(signal) {
    if (signal?.aborted) {
        const err = new Error('Operation aborted');
        err.name = 'AbortError';
        throw err;
    }
}

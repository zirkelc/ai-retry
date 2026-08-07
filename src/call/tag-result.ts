/**
 * Present an entry point's result as its tagged union member.
 *
 * A view rather than a copy, and deliberately so: the SDK's results expose most
 * of themselves through prototype getters, which neither a spread nor
 * `Object.assign` carries across — `{ ...result }` yields an object whose `text`
 * and `toolCalls` are silently `undefined`. Forwarding every read to the real
 * result is the only way to hand a condition what the caller would have got.
 *
 * The caller's own result is never touched, so nothing observes the tag except
 * a condition looking at this view.
 */
export function tagResult<RESULT extends object, OPERATION extends string>(
  operation: OPERATION,
  result: RESULT,
): { operation: OPERATION } & RESULT {
  return new Proxy(result, {
    get: (target, property) =>
      property === 'operation'
        ? operation
        : Reflect.get(target, property, target),
    has: (target, property) =>
      property === 'operation' || Reflect.has(target, property),
    ownKeys: (target) => [
      ...new Set<string | symbol>([...Reflect.ownKeys(target), 'operation']),
    ],
    getOwnPropertyDescriptor: (target, property) =>
      property === 'operation'
        ? { value: operation, enumerable: true, configurable: true }
        : Reflect.getOwnPropertyDescriptor(target, property),
  }) as { operation: OPERATION } & RESULT;
}

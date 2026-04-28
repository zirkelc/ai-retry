import { type AnyModel, type Condition } from './condition.js';
import { error } from './error.js';
import { or } from './or.js';

/**
 * A pattern accepted by `httpStatus`. Numbers match the response status
 * code; strings match the error message as a substring; regular
 * expressions match against both the stringified status code and the
 * error message.
 */
export type StatusPattern = number | string | RegExp;

/**
 * Match an `APICallError` by status code, message substring, or regular
 * expression. Numbers match the status code; strings match the message;
 * regular expressions match either the stringified status code or the
 * message. Mix any combination in a single call; matches when any
 * pattern matches.
 *
 * @example
 * httpStatus(529)
 * httpStatus(529, 'overloaded')
 * httpStatus(/^5\d\d$/)
 * httpStatus(529, 'overloaded', /rate.?limit/i)
 */
export function httpStatus<MODEL extends AnyModel = AnyModel>(
  ...patterns: Array<StatusPattern>
): Condition<MODEL> {
  const numbers = patterns.filter((p): p is number => typeof p === 'number');
  const strings = patterns.filter((p): p is string => typeof p === 'string');
  const regexes = patterns.filter((p): p is RegExp => p instanceof RegExp);

  const conditions: Array<Condition<MODEL>> = [];
  if (numbers.length || regexes.length) {
    conditions.push(error.statusCode<MODEL>(...numbers, ...regexes));
  }
  if (strings.length || regexes.length) {
    conditions.push(error.message<MODEL>(...strings, ...regexes));
  }
  return or(...conditions);
}

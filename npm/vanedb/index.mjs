// The unscoped name re-exports the scoped package at a matching version.
// `init` stays in the contract even though it is a no-op on Node: keeping it
// means this entry point can later dispatch to a native binding without a
// breaking change for anyone who wrote `await init()`.
export { default } from '@vanedb/wasm';
export * from '@vanedb/wasm';

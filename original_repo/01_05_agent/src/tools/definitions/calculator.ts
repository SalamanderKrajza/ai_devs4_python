/**
 * Calculator tool — simple math operations
 */
import type { Tool } from '../types.js'

type Operation = 'add' | 'subtract' | 'multiply' | 'divide'

interface CalculatorArgs {
  operation: Operation
  a: number
  b: number
}

function calculate({ operation, a, b }: CalculatorArgs): number {
  switch (operation) {
    case 'add': return a + b
    case 'subtract': return a - b
    case 'multiply': return a * b
    case 'divide': 
      if (b === 0) throw new Error('Division by zero')
      return a / b
    default:
      throw new Error(`Unknown operation: ${operation}`)
  }
}

export const calculatorTool: Tool = {
  type: 'sync',
  definition: {
    type: 'function',
    name: 'calculator',
    description: 'Perform basic math operations: add, subtract, multiply, divide',
    parameters: {
      type: 'object',
      properties: {
        operation: {
          type: 'string',
          enum: ['add', 'subtract', 'multiply', 'divide'],
          description: 'The math operation to perform',
        },
        a: {
          type: 'number',
          description: 'First operand',
        },
        b: {
          type: 'number',
          description: 'Second operand',
        },
      },
      required: ['operation', 'a', 'b'],
    },
  },
  handler: async (args) => {
    const { operation, a, b } = args as unknown as CalculatorArgs
    const result = calculate({ operation, a, b })
    return { ok: true, output: String(result) }
  },
}

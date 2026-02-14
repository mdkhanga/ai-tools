/**
 * REPL (Read-Eval-Print Loop) for the coding agent
 */

import * as readline from "node:readline";
import { stdin as input, stdout as output } from "node:process";
import { display } from "./display.js";

const VERSION = "0.1.0";

/** Built-in commands that the REPL handles directly */
const COMMANDS: Record<string, { description: string; handler: () => boolean }> = {
  "/help": {
    description: "Show available commands",
    handler: () => {
      display.section("Available Commands");
      for (const [cmd, { description }] of Object.entries(COMMANDS)) {
        console.log(`  ${cmd.padEnd(12)} - ${description}`);
      }
      display.newline();
      display.dim("Type any other text to chat with the agent.");
      return true; // continue REPL
    },
  },
  "/quit": {
    description: "Exit the agent",
    handler: () => {
      display.info("Goodbye!");
      return false; // exit REPL
    },
  },
  "/exit": {
    description: "Exit the agent (alias for /quit)",
    handler: () => COMMANDS["/quit"].handler(),
  },
};

/**
 * Handle user input - either a command or a message for the agent
 * Returns false if the REPL should exit
 */
function handleInput(input: string): boolean {
  const trimmed = input.trim();

  if (!trimmed) {
    return true; // empty input, continue
  }

  // Check for built-in commands
  if (trimmed.startsWith("/")) {
    const command = COMMANDS[trimmed.toLowerCase()];
    if (command) {
      return command.handler();
    } else {
      display.warn(`Unknown command: ${trimmed}`);
      display.dim("Type /help for available commands.");
      return true;
    }
  }

  // For now, echo the input (will be replaced with agent call in Task 4)
  display.newline();
  display.agent(`You said: ${trimmed}`);
  display.newline();

  return true;
}

/**
 * Start the interactive REPL
 */
export async function startRepl(): Promise<void> {
  display.banner(`mycoder-agent v${VERSION}`);
  display.newline();
  display.info("AI coding agent ready. Type /help for commands.");
  display.newline();

  const rl = readline.createInterface({
    input,
    output,
    prompt: "you> ",
  });

  rl.prompt();

  rl.on("line", (line) => {
    const shouldContinue = handleInput(line);
    if (shouldContinue) {
      rl.prompt();
    } else {
      rl.close();
    }
  });

  rl.on("close", () => {
    process.exit(0);
  });

  // Handle Ctrl+C gracefully
  rl.on("SIGINT", () => {
    display.newline();
    display.info("Interrupted. Goodbye!");
    rl.close();
  });
}

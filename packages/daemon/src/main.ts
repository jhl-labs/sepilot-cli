import { startDaemonRuntime } from './bootstrap.js'

/**
 * Foreground daemon entry. `argv` is the list *after* the program name and
 * after any `__daemon` dispatch token (i.e. what the daemon itself should see).
 */
export async function runDaemonMain(argv: string[] = process.argv.slice(2)): Promise<void> {
  const sub = argv[0]

  if (sub === 'memory-eval') {
    const { main: memoryEvalCommand } = await import('./memory/eval.js')
    await memoryEvalCommand(argv.slice(1))
    process.exit(process.exitCode ?? 0)
  }

  if (sub === 'init') {
    const { initCommand } = await import('./commands/init.js')
    await initCommand({ deviceName: argv[1], role: argv[2] })
    process.exit(0)
  }

  if (sub === 'doctor') {
    const { doctorCommand } = await import('./commands/doctor.js')
    await doctorCommand()
    process.exit(0)
  }

  const autoApproveCliFlag = argv.includes('--yes-to-everything')
  await startDaemonRuntime({
    autoApproveCliFlag,
    setupSignalHandlers: true,
    exitOnShutdown: true,
  })
}

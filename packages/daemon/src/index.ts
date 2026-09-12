import { runDaemonMain } from './main.js'

runDaemonMain(process.argv.slice(2)).catch((error) => {
  console.error(error)
  process.exit(1)
})

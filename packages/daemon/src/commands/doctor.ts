import { buildDoctorReport } from '../diagnostics/doctor.js'

export async function doctorCommand(options?: { dataDir?: string }): Promise<void> {
  const report = await buildDoctorReport({ dataDir: options?.dataDir })

  console.log('sepilotd Security Health Check')
  console.log('================================\n')

  for (const result of report.data) {
    console.log(`${statusLabel(result.status)} ${result.message}`)
  }

  console.log(`\nScore: ${report.summary.score.toFixed(1)}/10 (${report.summary.grade})`)
  console.log(`Warnings: ${report.summary.warnings}, Errors: ${report.summary.errors}`)

  if (report.summary.warnings > 0 || report.summary.errors > 0) {
    console.log('\nRecommendations:')
    for (const result of report.data) {
      if (result.status !== 'PASS') {
        console.log(`  [${result.status}] ${result.recommendation ?? result.message}`)
      }
    }
  }
}

function statusLabel(status: 'PASS' | 'WARN' | 'FAIL'): string {
  if (status === 'PASS') return '[PASS]'
  if (status === 'WARN') return '[WARN]'
  return '[FAIL]'
}

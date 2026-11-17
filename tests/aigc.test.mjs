// 网页 aigc 平面（对话主通路）的回归测试：网络全部打桩，不碰真账号。
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { WPSLingxiAuthPlugin, _internal } from '../index.mjs'

const COOKIE = `wps_sid=${'SID' + 'a'.repeat(29)}`
const SESSION = '/api/aigc/v3/assistant/sessions'

function frame(type, extra = {}) {
  return `event:execution\ndata:${JSON.stringify({ type, ...extra })}\n\n`
}
const endFrame = 'event:aigc\ndata:{"data":null,"type":"end"}\n\n'

// 打桩：建会话 → completions（帧序列由调用方给）→ DELETE 收尾。
async function chat({ body, frames, onCompletion }) {
  const plugin = await WPSLingxiAuthPlugin()
  const loader = await plugin.auth.loader(async () => ({ cookie: COOKIE }))
  const original = globalThis.fetch
  const completions = []
  globalThis.fetch = async (url, init = {}) => {
    const path = new URL(url).pathname
    if (path === SESSION && init.method === 'POST') return Response.json({ data: { session_id: 's-aigc' } })
    if (path === `${SESSION}/s-aigc/completions`) {
      const parsed = JSON.parse(init.body)
      completions.push({ headers: init.headers, body: parsed })
      onCompletion?.(parsed)
      return new Response(frames, { headers: { 'Content-Type': 'text/event-stream' } })
    }
    if (init.method === 'DELETE') return Response.json({})
    throw new Error(`unexpected request: ${url}`)
  }
  try {
    const res = await loader.fetch('https://mock/chat/completions', { body: JSON.stringify(body) })
    return { res, completions }
  } finally {
    globalThis.fetch = original
  }
}

function collectSse(text) {
  let content = ''
  let reasoning = ''
  const calls = []
  let finish = null
  for (const line of text.split('\n')) {
    if (!line.startsWith('data: ')) continue
    const payload = line.slice(6).trim()
    if (payload === '[DONE]') continue
    const value = JSON.parse(payload)
    assert.equal(value.error, undefined, '不应出现错误帧')
    const choice = value.choices?.[0]
    if (!choice) continue
    if (choice.finish_reason) finish = choice.finish_reason
    content += choice.delta?.content || ''
    reasoning += choice.delta?.reasoning_content || ''
    for (const call of choice.delta?.tool_calls || []) {
      const slot = (calls[call.index] ??= { name: '', arguments: '' })
      if (call.function?.name) slot.name = call.function.name
      if (call.function?.arguments) slot.arguments += call.function.arguments
    }
  }
  return { content, reasoning, calls, finish }
}

test('streaming text is translated from aigc frames to OpenAI SSE', async () => {
  const frames = [
    frame('user', { data: {} }),
    frame('text_start', { data: '' }),
    frame('text', { data: '你好' }),
    frame('text', { data: '，世界' }),
    frame('text_end', { data: '' }),
    endFrame,
  ].join('')
  const { res, completions } = await chat({
    body: { model: 'deepseek-v4.1-flash', messages: [{ role: 'user', content: '打个招呼' }] },
    frames,
    onCompletion: (body) => {
      assert.equal(body.expect_model, 'deepseek-v4.1-flash')
      assert.match(body.question, /User: 打个招呼/)
      assert.equal(typeof body.thinking, 'string')
    },
  })
  assert.equal(res.status, 200)
  assert.match(res.headers.get('content-type'), /text\/event-stream/)
  const result = collectSse(await res.text())
  assert.equal(result.content, '你好，世界')
  assert.equal(result.finish, 'stop')
  assert.equal(completions.length, 1)
})

test('non-stream folds the same frames into one completion', async () => {
  const frames = frame('text_start', { data: '' }) + frame('text', { data: 'OK' }) + frame('text_end', { data: '' }) + endFrame
  const { res } = await chat({
    body: { model: 'kimi-k3', stream: false, messages: [{ role: 'user', content: '只回复OK' }] },
    frames,
  })
  assert.equal(res.status, 200)
  const json = await res.json()
  assert.equal(json.choices[0].message.content, 'OK')
  assert.equal(json.choices[0].finish_reason, 'stop')
  assert.equal(json.model, 'kimi-k3')
})

test('the prompt flattens system, history and tools, and bridges tool calls back', async () => {
  const callPayload = '[{"name":"get_weather","parameters":{"city":"北京"}}]'
  const frames = [
    frame('text_start', { data: '' }),
    frame('text', { data: `<|FunctionCallBegin|>${callPayload}<|FunctionCallEnd|>` }),
    frame('text_end', { data: '' }),
    endFrame,
  ].join('')
  const { res, completions } = await chat({
    body: {
      model: 'deepseek-v4.1-flash',
      messages: [
        { role: 'system', content: '保持简洁' },
        { role: 'user', content: '北京天气？' },
        { role: 'assistant', content: '我查一下' },
        { role: 'user', content: '快点' },
      ],
      tools: [{ type: 'function', function: { name: 'get_weather', description: '查天气', parameters: { type: 'object', properties: { city: { type: 'string' } } } } }],
    },
    frames,
  })
  const question = completions[0].body.question
  assert.match(question, /SYSTEM:\n保持简洁/)
  assert.match(question, /TOOLS:/)
  assert.match(question, /get_weather/)
  assert.match(question, /User: 北京天气？/)
  assert.match(question, /Assistant: 我查一下/)
  assert.match(question, /User: 快点$/)

  const result = collectSse(await res.text())
  assert.equal(result.content, '')
  assert.equal(result.finish, 'tool_calls')
  assert.deepEqual(result.calls.map((c) => c.name), ['get_weather'])
  assert.deepEqual(JSON.parse(result.calls[0].arguments), { city: '北京' })
})

test('upstream error frames surface with the upstream code and status', async () => {
  const frames = 'event:error\ndata:{"result":"PromptTokenExceedLimit","msg":"场景token超过限制"}\n\n'
  const { res } = await chat({
    body: { model: 'deepseek-v4.1-flash', stream: false, messages: [{ role: 'user', content: 'hi' }] },
    frames,
  })
  assert.equal(res.status, 400)
  const json = await res.json()
  assert.equal(json.error.code, 'PromptTokenExceedLimit')
  assert.match(json.error.message, /token/)
})

test('a prompt over the byte budget is refused before any upstream call', () => {
  const huge = 'x'.repeat(512 * 1024)
  assert.throws(
    () => _internal.aigcQuestion({ messages: [{ role: 'user', content: huge }] }, []),
    (error) => error.code === 'prompt_token_exceed_limit' && error.status === 400,
  )
})

// GLM 方言：参数值嵌同名标签/引号/&& 时非贪婪正则会截断（修复记录里的"错版"）。
test('GLM parameter-shell dialect with nested tags parses into tool calls', () => {
  const parser = _internal.createToolCallParser(['terminal'])
  const events = [
    ...parser.push('<|FunctionCallBegin|><parameter name="terminal"><parameter name="command">ls -la "my folder" && echo done</parameter><parameter name="background" string="false">true</parameter></parameter><|FunctionCallEnd|>'),
    ...parser.flush(),
  ]
  assert.equal(events.length, 1)
  assert.equal(events[0].type, 'calls')
  assert.equal(events[0].calls[0].name, 'terminal')
  const args = JSON.parse(events[0].calls[0].arguments)
  assert.equal(args.command, 'ls -la "my folder" && echo done')
  assert.equal(args.background, true)
})

test('reasoning effort accepts medium and minimal like magpie sends', async () => {
  const frames = frame('text_start', { data: '' }) + frame('text', { data: 'ok' }) + frame('text_end', { data: '' }) + endFrame
  for (const effort of ['medium', 'minimal', 'high']) {
    const { completions } = await chat({
      body: { model: 'deepseek-v4.1-flash', stream: false, reasoning_effort: effort, messages: [{ role: 'user', content: 'hi' }] },
      frames,
    })
    assert.ok(['0', '1', '2'].includes(completions[0].body.reasoning_effort), `effort ${effort} should map to 0/1/2`)
  }
})

// 参考 inceptionlabs2api：模型会照训练数据吐 Hermes/Anthropic 信封，认下来。
test('drift envelopes <tool_call> and <tool_calls><invoke> parse', () => {
  const json = _internal.createToolCallParser(['get_weather'])
  const events = [
    ...json.push('<tool_call>{"name":"get_weather","arguments":{"city":"上海"}}</tool_call>'),
    ...json.flush(),
  ]
  assert.equal(events[0].type, 'calls')
  assert.deepEqual(JSON.parse(events[0].calls[0].arguments), { city: '上海' })

  const xml = _internal.createToolCallParser(['get_weather'])
  const xmlEvents = [
    ...xml.push('<tool_calls><invoke name="get_weather"><parameter name="city">北京</parameter></invoke></tool_calls>'),
    ...xml.flush(),
  ]
  assert.equal(xmlEvents[0].type, 'calls')
  assert.equal(xmlEvents[0].calls[0].name, 'get_weather')
  assert.deepEqual(JSON.parse(xmlEvents[0].calls[0].arguments), { city: '北京' })
})

// 上游被 token 上限截断：本地补齐括号/引号，别把半截信封吐回正文。
test('a truncated JSON call block is repaired before falling back to prose', () => {
  const parser = _internal.createToolCallParser(['get_weather'])
  const events = [
    ...parser.push('<|FunctionCallBegin|>[{"name":"get_weather","parameters":{"city":"北'),
    ...parser.flush(),
  ]
  assert.equal(events.length, 1)
  assert.equal(events[0].type, 'calls')
  assert.deepEqual(JSON.parse(events[0].calls[0].arguments), { city: '北' })
})

test('the tool prompt carries a concrete example and the continuation rule', () => {
  const question = _internal.aigcQuestion({
    tool_choice: 'required',
    messages: [{ role: 'user', content: '北京天气？' }],
    tools: [{ type: 'function', function: { name: 'get_weather', parameters: { type: 'object', required: ['city'], properties: { city: { type: 'string' } } } } }],
  }, [{
    name: 'get_weather',
    description: '查天气',
    parameters: { type: 'object', required: ['city'], properties: { city: { type: 'string' } } },
  }])
  assert.match(question, /示例（get_weather）：<\|FunctionCallBegin\|>\[\{"name":"get_weather","parameters":\{"city":""\}\}\]/)
  assert.match(question, /立刻发起下一个工具调用/)
  assert.match(question, /本轮必须至少调用一个工具/)
})

test('a tool result is labelled by tool_call_id lookup and ends with the continuation nudge', () => {
  const question = _internal.aigcQuestion({
    messages: [
      { role: 'user', content: '北京天气？' },
      { role: 'assistant', content: '', tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'get_weather', arguments: '{"city":"北京"}' } }] },
      { role: 'tool', tool_call_id: 'call_1', content: '晴 25℃' },
    ],
  }, [])
  assert.match(question, /Tool \(get_weather\): 晴 25℃/)
  assert.match(question, /立刻发起下一个工具调用/)
})

// ── 坏轮次重问：提示词桥没有原生保证，空答/坏信封/强制零调用都重问一次 ──────────

const TOOLS = [{ type: 'function', function: { name: 'get_weather', parameters: { type: 'object', required: ['city'], properties: { city: { type: 'string' } } } } }]
const CALL_FRAMES = frame('text_start', { data: '' })
  + frame('text', { data: '<|FunctionCallBegin|>[{"name":"get_weather","parameters":{"city":"北京"}}]<|FunctionCallEnd|>' })
  + frame('text_end', { data: '' }) + endFrame
// 空答：有完整帧但没有文本。
const EMPTY_FRAMES = frame('text_start', { data: '' }) + frame('text_end', { data: '' }) + endFrame

// 每一轮 completions 返回不同的帧序列。
async function chatRounds({ body, rounds }) {
  const plugin = await WPSLingxiAuthPlugin()
  const loader = await plugin.auth.loader(async () => ({ cookie: COOKIE }))
  const original = globalThis.fetch
  const completions = []
  globalThis.fetch = async (url, init = {}) => {
    const path = new URL(url).pathname
    if (path === SESSION && init.method === 'POST') return Response.json({ data: { session_id: `s-${completions.length}` } })
    if (path.endsWith('/completions')) {
      completions.push(JSON.parse(init.body))
      return new Response(rounds[Math.min(completions.length - 1, rounds.length - 1)], { headers: { 'Content-Type': 'text/event-stream' } })
    }
    if (init.method === 'DELETE') return Response.json({})
    throw new Error(`unexpected request: ${url}`)
  }
  try {
    const res = await loader.fetch('https://mock/chat/completions', { body: JSON.stringify(body) })
    // 流式重问发生在流尾：必须先把 body 读完，stub 才能卸载。
    const text = await res.text()
    return { status: res.status, text, completions }
  } finally {
    globalThis.fetch = original
  }
}

test('an empty tool round is retried once with a nudge', async () => {
  const { text, completions } = await chatRounds({
    body: { model: 'deepseek-v4.1-flash', stream: true, messages: [{ role: 'user', content: '天气？' }], tools: TOOLS },
    rounds: [EMPTY_FRAMES, CALL_FRAMES],
  })
  const result = collectSse(text)
  assert.equal(completions.length, 2)
  assert.match(completions[1].question, /上一条回复是空的/)
  assert.equal(result.finish, 'tool_calls')
  assert.deepEqual(JSON.parse(result.calls[0].arguments), { city: '北京' })
})

test('a broken call envelope is retried and never leaks into the reply', async () => {
  const broken = frame('text_start', { data: '' })
    + frame('text', { data: '<|FunctionCallBegin|>{"name": "get_weather"' })
    + frame('text_end', { data: '' }) + endFrame
  const { text, completions } = await chatRounds({
    body: { model: 'deepseek-v4.1-flash', stream: true, messages: [{ role: 'user', content: '天气？' }], tools: TOOLS },
    rounds: [broken, CALL_FRAMES],
  })
  const result = collectSse(text)
  assert.equal(completions.length, 2)
  assert.match(completions[1].question, /格式无法解析/)
  assert.equal(result.finish, 'tool_calls')
  assert.doesNotMatch(result.content, /FunctionCallBegin/)
})

test('clean prose is a legal terminal and is not retried; forced tool_choice is', async () => {
  const prose = frame('text_start', { data: '' }) + frame('text', { data: '直接回答。' }) + frame('text_end', { data: '' }) + endFrame
  const clean = await chatRounds({
    body: { model: 'deepseek-v4.1-flash', messages: [{ role: 'user', content: '你好' }], tools: TOOLS },
    rounds: [prose],
  })
  assert.equal(clean.completions.length, 1)
  assert.equal(collectSse(clean.text).content, '直接回答。')

  // 流式下正文已推出去就收不回来，不能再重问：原样放行。
  const forcedStream = await chatRounds({
    body: { model: 'deepseek-v4.1-flash', tool_choice: 'required', messages: [{ role: 'user', content: '你好' }], tools: TOOLS },
    rounds: [prose, CALL_FRAMES],
  })
  assert.equal(forcedStream.completions.length, 1)
  assert.equal(collectSse(forcedStream.text).finish, 'stop')

  // 非流式本来就要等整轮，强制调用没调用就重问一次。
  const forced = await chatRounds({
    body: { model: 'deepseek-v4.1-flash', stream: false, tool_choice: 'required', messages: [{ role: 'user', content: '你好' }], tools: TOOLS },
    rounds: [prose, CALL_FRAMES],
  })
  assert.equal(forced.completions.length, 2)
  assert.match(forced.completions[1].question, /必须调用 一个工具/)
  assert.equal(JSON.parse(forced.text).choices[0].finish_reason, 'tool_calls')
})

test('a failed retry keeps the original round instead of erroring', async () => {
  const { status, text, completions } = await chatRounds({
    body: { model: 'deepseek-v4.1-flash', stream: false, messages: [{ role: 'user', content: '天气？' }], tools: TOOLS },
    rounds: [EMPTY_FRAMES, 'event:error\ndata:{"result":"RateLimiter","msg":"限流"}\n\n'],
  })
  assert.equal(completions.length, 2)
  assert.equal(status, 200)
  assert.equal(JSON.parse(text).choices[0].finish_reason, 'stop')
})

test('plain angle brackets in text never parse as tool calls', () => {
  const parser = _internal.createToolCallParser(['get_weather'])
  const events = [
    ...parser.push('文本 <div>不是工具</div> <parameter name="x">示例</parameter>'),
    ...parser.flush(),
  ]
  for (const event of events) assert.equal(event.type, 'text')
})

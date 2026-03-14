Getting Started
Using Hono is super easy. We can set up the project, write code, develop with a local server, and deploy quickly. The same code will work on any runtime, just with different entry points. Let's look at the basic usage of Hono.

Starter
Starter templates are available for each platform. Use the following "create-hono" command.


npm

yarn

pnpm

bun

deno

npm create hono@latest my-app
Then you will be asked which template you would like to use. Let's select Cloudflare Workers for this example.


? Which template do you want to use?
    aws-lambda
    bun
    cloudflare-pages
❯   cloudflare-workers
    deno
    fastly
    nextjs
    nodejs
    vercel
The template will be pulled into my-app, so go to it and install the dependencies.


npm

yarn

pnpm

bun

cd my-app
npm i
Once the package installation is complete, run the following command to start up a local server.


npm

yarn

pnpm

bun

npm run dev
Hello World
You can write code in TypeScript with the Cloudflare Workers development tool "Wrangler", Deno, Bun, or others without being aware of transpiling.

Write your first application with Hono in src/index.ts. The example below is a starter Hono application.

The import and the final export default parts may vary from runtime to runtime, but all of the application code will run the same code everywhere.


import { Hono } from 'hono'

const app = new Hono()

app.get('/', (c) => {
  return c.text('Hello Hono!')
})

export default app
Start the development server and access http://localhost:8787 with your browser.


npm

yarn

pnpm

bun

npm run dev
Return JSON
Returning JSON is also easy. The following is an example of handling a GET Request to /api/hello and returning an application/json Response.


app.get('/api/hello', (c) => {
  return c.json({
    ok: true,
    message: 'Hello Hono!',
  })
})
Request and Response
Getting a path parameter, URL query value, and appending a Response header is written as follows.


app.get('/posts/:id', (c) => {
  const page = c.req.query('page')
  const id = c.req.param('id')
  c.header('X-Message', 'Hi!')
  return c.text(`You want to see ${page} of ${id}`)
})
We can easily handle POST, PUT, and DELETE not only GET.


app.post('/posts', (c) => c.text('Created!', 201))
app.delete('/posts/:id', (c) =>
  c.text(`${c.req.param('id')} is deleted!`)
)
Return HTML
You can write HTML with the html Helper or using JSX syntax. If you want to use JSX, rename the file to src/index.tsx and configure it (check with each runtime as it is different). Below is an example using JSX.


const View = () => {
  return (
    <html>
      <body>
        <h1>Hello Hono!</h1>
      </body>
    </html>
  )
}

app.get('/page', (c) => {
  return c.html(<View />)
})
Return raw Response
You can also return the raw Response.


app.get('/', () => {
  return new Response('Good morning!')
})
Using Middleware
Middleware can do the hard work for you. For example, add in Basic Authentication.


import { basicAuth } from 'hono/basic-auth'

// ...

app.use(
  '/admin/*',
  basicAuth({
    username: 'admin',
    password: 'secret',
  })
)

app.get('/admin', (c) => {
  return c.text('You are authorized!')
})
There are useful built-in middleware including Bearer and authentication using JWT, CORS and ETag. Hono also provides third-party middleware using external libraries such as GraphQL Server and Firebase Auth. And, you can make your own middleware.

Adapter
There are Adapters for platform-dependent functions, e.g., handling static files or WebSocket. For example, to handle WebSocket in Cloudflare Workers, import hono/cloudflare-workers.


import { upgradeWebSocket } from 'hono/cloudflare-workers'

app.get(
  '/ws',
  upgradeWebSocket((c) => {
    // ...
  })
)
Next step
Most code will work on any platform, but there are guides for each. For instance, how to set up projects or how to deploy. Please see the page for the exact platform you want to use to create your application!Node.js
Node.js is an open-source, cross-platform JavaScript runtime environment.

Hono was not designed for Node.js at first. But with a Node.js Adapter it can run on Node.js as well.

INFO

It works on Node.js versions greater than 18.x. The specific required Node.js versions are as follows:

18.x => 18.14.1+
19.x => 19.7.0+
20.x => 20.0.0+
Essentially, you can simply use the latest version of each major release.

1. Setup
A starter for Node.js is available. Start your project with "create-hono" command. Select nodejs template for this example.


npm

yarn

pnpm

bun

deno

npm create hono@latest my-app
Move to my-app and install the dependencies.


npm

yarn

pnpm

bun

cd my-app
npm i
2. Hello World
Edit src/index.ts:


import { serve } from '@hono/node-server'
import { Hono } from 'hono'

const app = new Hono()
app.get('/', (c) => c.text('Hello Node.js!'))

serve(app)
If you want to gracefully shut down the server, write it like this:


const server = serve(app)

// graceful shutdown
process.on('SIGINT', () => {
  server.close()
  process.exit(0)
})
process.on('SIGTERM', () => {
  server.close((err) => {
    if (err) {
      console.error(err)
      process.exit(1)
    }
    process.exit(0)
  })
})
3. Run
Run the development server locally. Then, access http://localhost:3000 in your Web browser.


npm

yarn

pnpm

npm run dev
Change port number
You can specify the port number with the port option.


serve({
  fetch: app.fetch,
  port: 8787,
})
Access the raw Node.js APIs
You can access the Node.js APIs from c.env.incoming and c.env.outgoing.


import { Hono } from 'hono'
import { serve, type HttpBindings } from '@hono/node-server'
// or `Http2Bindings` if you use HTTP2

type Bindings = HttpBindings & {
  /* ... */
}

const app = new Hono<{ Bindings: Bindings }>()

app.get('/', (c) => {
  return c.json({
    remoteAddress: c.env.incoming.socket.remoteAddress,
  })
})

serve(app)
Serve static files
You can use serveStatic to serve static files from the local file system. For example, suppose the directory structure is as follows:


./
├── favicon.ico
├── index.ts
└── static
    ├── hello.txt
    └── image.png
If a request to the path /static/* comes in and you want to return a file under ./static, you can write the following:


import { serveStatic } from '@hono/node-server/serve-static'

app.use('/static/*', serveStatic({ root: './' }))
Use the path option to serve favicon.ico in the directory root:


app.use('/favicon.ico', serveStatic({ path: './favicon.ico' }))
If a request to the path /hello.txt or /image.png comes in and you want to return a file named ./static/hello.txt or ./static/image.png, you can use the following:


app.use('*', serveStatic({ root: './static' }))
rewriteRequestPath
If you want to map http://localhost:3000/static/* to ./statics, you can use the rewriteRequestPath option:


app.get(
  '/static/*',
  serveStatic({
    root: './',
    rewriteRequestPath: (path) =>
      path.replace(/^\/static/, '/statics'),
  })
)
http2
You can run hono on a Node.js http2 Server.

unencrypted http2

import { createServer } from 'node:http2'

const server = serve({
  fetch: app.fetch,
  createServer,
})
encrypted http2

import { createSecureServer } from 'node:http2'
import { readFileSync } from 'node:fs'

const server = serve({
  fetch: app.fetch,
  createServer: createSecureServer,
  serverOptions: {
    key: readFileSync('localhost-privkey.pem'),
    cert: readFileSync('localhost-cert.pem'),
  },
})
Building & Deployment

npm

yarn

pnpm

bun

npm run build
INFO

Apps with a front-end framework may need to use Hono's Vite plugins.

Dockerfile
Here is an example of a nodejs Dockerfile.


FROM node:22-alpine AS base

FROM base AS builder

RUN apk add --no-cache gcompat
WORKDIR /app

COPY package*json tsconfig.json src ./

RUN npm ci && \
    npm run build && \
    npm prune --production

FROM base AS runner
WORKDIR /app

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 hono

COPY --from=builder --chown=hono:nodejs /app/node_modules /app/node_modules
COPY --from=builder --chown=hono:nodejs /app/dist /app/dist
COPY --from=builder --chown=hono:nodejs /app/package.json /app/package.json

USER hono
EXPOSE 3000

CMD ["node", "/app/dist/index.js"]
App - Hono
Hono is the primary object. It will be imported first and used until the end.


import { Hono } from 'hono'

const app = new Hono()
//...

export default app // for Cloudflare Workers or Bun
Methods
An instance of Hono has the following methods.

app.HTTP_METHOD([path,]handler|middleware...)
app.all([path,]handler|middleware...)
app.on(method|method[], path|path[], handler|middleware...)
app.use([path,]middleware)
app.route(path, [app])
app.basePath(path)
app.notFound(handler)
app.onError(err, handler)
app.mount(path, anotherApp)
app.fire()
app.fetch(request, env, event)
app.request(path, options)
The first part of them is used for routing, please refer to the routing section.

Not Found
app.notFound allows you to customize a Not Found Response.


app.notFound((c) => {
  return c.text('Custom 404 Message', 404)
})
WARNING

The notFound method is only called from the top-level app. For more information, see this issue.

Error Handling
app.onError allows you to handle uncaught errors and return a custom Response.


app.onError((err, c) => {
  console.error(`${err}`)
  return c.text('Custom Error Message', 500)
})
INFO

If both a parent app and its routes have onError handlers, the route-level handlers get priority.

fire()
WARNING

app.fire() is deprecated. Use fire() from hono/service-worker instead. See the Service Worker documentation for details.

app.fire() automatically adds a global fetch event listener.

This can be useful for environments that adhere to the Service Worker API, such as non-ES module Cloudflare Workers.

app.fire() executes the following for you:


addEventListener('fetch', (event: FetchEventLike): void => {
  event.respondWith(this.dispatch(...))
})
fetch()
app.fetch will be entry point of your application.

For Cloudflare Workers, you can use the following:


export default {
  fetch(request: Request, env: Env, ctx: ExecutionContext) {
    return app.fetch(request, env, ctx)
  },
}
or just do:


export default app
Bun:


export default app 
export default {  
  port: 3000, 
  fetch: app.fetch, 
} 
request()
request is a useful method for testing.

You can pass a URL or pathname to send a GET request. app will return a Response object.


test('GET /hello is ok', async () => {
  const res = await app.request('/hello')
  expect(res.status).toBe(200)
})
You can also pass a Request object:


test('POST /message is ok', async () => {
  const req = new Request('Hello!', {
    method: 'POST',
  })
  const res = await app.request(req)
  expect(res.status).toBe(201)
})
mount()
The mount() allows you to mount applications built with other frameworks into your Hono application.


import { Router as IttyRouter } from 'itty-router'
import { Hono } from 'hono'

// Create itty-router application
const ittyRouter = IttyRouter()

// Handle `GET /itty-router/hello`
ittyRouter.get('/hello', () => new Response('Hello from itty-router'))

// Hono application
const app = new Hono()

// Mount!
app.mount('/itty-router', ittyRouter.handle)
strict mode
Strict mode defaults to true and distinguishes the following routes.

/hello
/hello/
app.get('/hello') will not match GET /hello/.

By setting strict mode to false, both paths will be treated equally.


const app = new Hono({ strict: false })
router option
The router option specifies which router to use. The default router is SmartRouter. If you want to use RegExpRouter, pass it to a new Hono instance:


import { RegExpRouter } from 'hono/router/reg-exp-router'

const app = new Hono({ router: new RegExpRouter() })
Generics
You can pass Generics to specify the types of Cloudflare Workers Bindings and variables used in c.set/c.get.


type Bindings = {
  TOKEN: string
}

type Variables = {
  user: User
}

const app = new Hono<{
  Bindings: Bindings
  Variables: Variables
}>()

app.use('/auth/*', async (c, next) => {
  const token = c.env.TOKEN // token is `string`
  // ...
  c.set('user', user) // user should be `User`
  await next()
})Routing
Routing of Hono is flexible and intuitive. Let's take a look.

Basic

// HTTP Methods
app.get('/', (c) => c.text('GET /'))
app.post('/', (c) => c.text('POST /'))
app.put('/', (c) => c.text('PUT /'))
app.delete('/', (c) => c.text('DELETE /'))

// Wildcard
app.get('/wild/*/card', (c) => {
  return c.text('GET /wild/*/card')
})

// Any HTTP methods
app.all('/hello', (c) => c.text('Any Method /hello'))

// Custom HTTP method
app.on('PURGE', '/cache', (c) => c.text('PURGE Method /cache'))

// Multiple Method
app.on(['PUT', 'DELETE'], '/post', (c) =>
  c.text('PUT or DELETE /post')
)

// Multiple Paths
app.on('GET', ['/hello', '/ja/hello', '/en/hello'], (c) =>
  c.text('Hello')
)
Path Parameter

app.get('/user/:name', async (c) => {
  const name = c.req.param('name')
  // ...
})
or all parameters at once:


app.get('/posts/:id/comment/:comment_id', async (c) => {
  const { id, comment_id } = c.req.param()
  // ...
})
Optional Parameter

// Will match `/api/animal` and `/api/animal/:type`
app.get('/api/animal/:type?', (c) => c.text('Animal!'))
Regexp

app.get('/post/:date{[0-9]+}/:title{[a-z]+}', async (c) => {
  const { date, title } = c.req.param()
  // ...
})
Including slashes

app.get('/posts/:filename{.+\\.png}', async (c) => {
  //...
})
Chained route

app
  .get('/endpoint', (c) => {
    return c.text('GET /endpoint')
  })
  .post((c) => {
    return c.text('POST /endpoint')
  })
  .delete((c) => {
    return c.text('DELETE /endpoint')
  })
Grouping
You can group the routes with the Hono instance and add them to the main app with the route method.


const book = new Hono()

book.get('/', (c) => c.text('List Books')) // GET /book
book.get('/:id', (c) => {
  // GET /book/:id
  const id = c.req.param('id')
  return c.text('Get Book: ' + id)
})
book.post('/', (c) => c.text('Create Book')) // POST /book

const app = new Hono()
app.route('/book', book)
Grouping without changing base
You can also group multiple instances while keeping base.


const book = new Hono()
book.get('/book', (c) => c.text('List Books')) // GET /book
book.post('/book', (c) => c.text('Create Book')) // POST /book

const user = new Hono().basePath('/user')
user.get('/', (c) => c.text('List Users')) // GET /user
user.post('/', (c) => c.text('Create User')) // POST /user

const app = new Hono()
app.route('/', book) // Handle /book
app.route('/', user) // Handle /user
Base path
You can specify the base path.


const api = new Hono().basePath('/api')
api.get('/book', (c) => c.text('List Books')) // GET /api/book
Routing with hostname
It works fine if it includes a hostname.


const app = new Hono({
  getPath: (req) => req.url.replace(/^https?:\/([^?]+).*$/, '$1'),
})

app.get('/www1.example.com/hello', (c) => c.text('hello www1'))
app.get('/www2.example.com/hello', (c) => c.text('hello www2'))
Routing with host Header value
Hono can handle the host header value if you set the getPath() function in the Hono constructor.


const app = new Hono({
  getPath: (req) =>
    '/' +
    req.headers.get('host') +
    req.url.replace(/^https?:\/\/[^/]+(\/[^?]*).*/, '$1'),
})

app.get('/www1.example.com/hello', (c) => c.text('hello www1'))

// A following request will match the route:
// new Request('http://www1.example.com/hello', {
//  headers: { host: 'www1.example.com' },
// })
By applying this, for example, you can change the routing by User-Agent header.

Routing priority
Handlers or middleware will be executed in registration order.


app.get('/book/a', (c) => c.text('a')) // a
app.get('/book/:slug', (c) => c.text('common')) // common

GET /book/a ---> `a`
GET /book/b ---> `common`
When a handler is executed, the process will be stopped.


app.get('*', (c) => c.text('common')) // common
app.get('/foo', (c) => c.text('foo')) // foo

GET /foo ---> `common` // foo will not be dispatched
If you have the middleware that you want to execute, write the code above the handler.


app.use(logger())
app.get('/foo', (c) => c.text('foo'))
If you want to have a "fallback" handler, write the code below the other handler.


app.get('/bar', (c) => c.text('bar')) // bar
app.get('*', (c) => c.text('fallback')) // fallback

GET /bar ---> `bar`
GET /foo ---> `fallback`
Grouping ordering
Note that the mistake of grouping routings is hard to notice. The route() function takes the stored routing from the second argument (such as three or two) and adds it to its own (two or app) routing.


three.get('/hi', (c) => c.text('hi'))
two.route('/three', three)
app.route('/two', two)

export default app
It will return 200 response.


GET /two/three/hi ---> `hi`
However, if they are in the wrong order, it will return a 404.


three.get('/hi', (c) => c.text('hi'))
app.route('/two', two) // `two` does not have routes
two.route('/three', three)

export default app

GET /two/three/hi ---> 404 Not Found
Edit this page on GitHub
Context
The Context object is instantiated for each request and kept until the response is returned. You can put values in it, set headers and a status code you want to return, and access HonoRequest and Response objects.

req
req is an instance of HonoRequest. For more details, see HonoRequest.


app.get('/hello', (c) => {
  const userAgent = c.req.header('User-Agent')
  // ...
})
status()
You can set an HTTP status code with c.status(). The default is 200. You don't have to use c.status() if the code is 200.


app.post('/posts', (c) => {
  // Set HTTP status code
  c.status(201)
  return c.text('Your post is created!')
})
header()
You can set HTTP Headers for the response.


app.get('/', (c) => {
  // Set headers
  c.header('X-Message', 'My custom message')
  return c.text('HellO!')
})
body()
Return an HTTP response.

INFO

Note: When returning text or HTML, it is recommended to use c.text() or c.html().


app.get('/welcome', (c) => {
  c.header('Content-Type', 'text/plain')
  // Return the response body
  return c.body('Thank you for coming')
})
You can also write the following.


app.get('/welcome', (c) => {
  return c.body('Thank you for coming', 201, {
    'X-Message': 'Hello!',
    'Content-Type': 'text/plain',
  })
})
The response is the same Response object as below.


new Response('Thank you for coming', {
  status: 201,
  headers: {
    'X-Message': 'Hello!',
    'Content-Type': 'text/plain',
  },
})
text()
Render text as Content-Type:text/plain.


app.get('/say', (c) => {
  return c.text('Hello!')
})
json()
Render JSON as Content-Type:application/json.


app.get('/api', (c) => {
  return c.json({ message: 'Hello!' })
})
html()
Render HTML as Content-Type:text/html.


app.get('/', (c) => {
  return c.html('<h1>Hello! Hono!</h1>')
})
notFound()
Return a Not Found Response. You can customize it with app.notFound().


app.get('/notfound', (c) => {
  return c.notFound()
})
redirect()
Redirect, default status code is 302.


app.get('/redirect', (c) => {
  return c.redirect('/')
})
app.get('/redirect-permanently', (c) => {
  return c.redirect('/', 301)
})
res
You can access the Response object that will be returned.


// Response object
app.use('/', async (c, next) => {
  await next()
  c.res.headers.append('X-Debug', 'Debug message')
})
set() / get()
Get and set arbitrary key-value pairs, with a lifetime of the current request. This allows passing specific values between middleware or from middleware to route handlers.


app.use(async (c, next) => {
  c.set('message', 'Hono is cool!!')
  await next()
})

app.get('/', (c) => {
  const message = c.get('message')
  return c.text(`The message is "${message}"`)
})
Pass the Variables as Generics to the constructor of Hono to make it type-safe.


type Variables = {
  message: string
}

const app = new Hono<{ Variables: Variables }>()
The value of c.set / c.get are retained only within the same request. They cannot be shared or persisted across different requests.

var
You can also access the value of a variable with c.var.


const result = c.var.client.oneMethod()
If you want to create the middleware which provides a custom method, write like the following:


type Env = {
  Variables: {
    echo: (str: string) => string
  }
}

const app = new Hono()

const echoMiddleware = createMiddleware<Env>(async (c, next) => {
  c.set('echo', (str) => str)
  await next()
})

app.get('/echo', echoMiddleware, (c) => {
  return c.text(c.var.echo('Hello!'))
})
If you want to use the middleware in multiple handlers, you can use app.use(). Then, you have to pass the Env as Generics to the constructor of Hono to make it type-safe.


const app = new Hono<Env>()

app.use(echoMiddleware)

app.get('/echo', (c) => {
  return c.text(c.var.echo('Hello!'))
})
render() / setRenderer()
You can set a layout using c.setRenderer() within a custom middleware.


app.use(async (c, next) => {
  c.setRenderer((content) => {
    return c.html(
      <html>
        <body>
          <p>{content}</p>
        </body>
      </html>
    )
  })
  await next()
})
Then, you can utilize c.render() to create responses within this layout.


app.get('/', (c) => {
  return c.render('Hello!')
})
The output of which will be:


<html>
  <body>
    <p>Hello!</p>
  </body>
</html>
Additionally, this feature offers the flexibility to customize arguments. To ensure type safety, types can be defined as:


declare module 'hono' {
  interface ContextRenderer {
    (
      content: string | Promise<string>,
      head: { title: string }
    ): Response | Promise<Response>
  }
}
Here's an example of how you can use this:


app.use('/pages/*', async (c, next) => {
  c.setRenderer((content, head) => {
    return c.html(
      <html>
        <head>
          <title>{head.title}</title>
        </head>
        <body>
          <header>{head.title}</header>
          <p>{content}</p>
        </body>
      </html>
    )
  })
  await next()
})

app.get('/pages/my-favorite', (c) => {
  return c.render(<p>Ramen and Sushi</p>, {
    title: 'My favorite',
  })
})

app.get('/pages/my-hobbies', (c) => {
  return c.render(<p>Watching baseball</p>, {
    title: 'My hobbies',
  })
})
executionCtx
You can access Cloudflare Workers' specific ExecutionContext.


// ExecutionContext object
app.get('/foo', async (c) => {
  c.executionCtx.waitUntil(c.env.KV.put(key, data))
  // ...
})
event
You can access Cloudflare Workers' specific FetchEvent. This was used in "Service Worker" syntax. But, it is not recommended now.


// Type definition to make type inference
type Bindings = {
  MY_KV: KVNamespace
}

const app = new Hono<{ Bindings: Bindings }>()

// FetchEvent object (only set when using Service Worker syntax)
app.get('/foo', async (c) => {
  c.event.waitUntil(c.env.MY_KV.put(key, data))
  // ...
})
env
In Cloudflare Workers Environment variables, secrets, KV namespaces, D1 database, R2 bucket etc. that are bound to a worker are known as bindings. Regardless of type, bindings are always available as global variables and can be accessed via the context c.env.BINDING_KEY.


// Type definition to make type inference
type Bindings = {
  MY_KV: KVNamespace
}

const app = new Hono<{ Bindings: Bindings }>()

// Environment object for Cloudflare Workers
app.get('/', async (c) => {
  c.env.MY_KV.get('my-key')
  // ...
})
error
If the Handler throws an error, the error object is placed in c.error. You can access it in your middleware.


app.use(async (c, next) => {
  await next()
  if (c.error) {
    // do something...
  }
})
ContextVariableMap
For instance, if you wish to add type definitions to variables when a specific middleware is used, you can extend ContextVariableMap. For example:


declare module 'hono' {
  interface ContextVariableMap {
    result: string
  }
}
You can then utilize this in your middleware:


const mw = createMiddleware(async (c, next) => {
  c.set('result', 'some values') // result is a string
  await next()
})
In a handler, the variable is inferred as the proper type:


app.get('/', (c) => {
  const val = c.get('result') // val is a string
  // ...
  return c.json({ result: val })
})HonoRequest
The HonoRequest is an object that can be taken from c.req which wraps a Request object.

param()
Get the values of path parameters.


// Captured params
app.get('/entry/:id', async (c) => {
  const id = c.req.param('id')
  // ...
})

// Get all params at once
app.get('/entry/:id/comment/:commentId', async (c) => {
  const { id, commentId } = c.req.param()
})
query()
Get querystring parameters.


// Query params
app.get('/search', async (c) => {
  const query = c.req.query('q')
})

// Get all params at once
app.get('/search', async (c) => {
  const { q, limit, offset } = c.req.query()
})
queries()
Get multiple querystring parameter values, e.g. /search?tags=A&tags=B


app.get('/search', async (c) => {
  // tags will be string[]
  const tags = c.req.queries('tags')
  // ...
})
header()
Get the request header value.


app.get('/', (c) => {
  const userAgent = c.req.header('User-Agent')
  return c.text(`Your user agent is ${userAgent}`)
})
WARNING

When c.req.header() is called with no arguments, all keys in the returned record are lowercase.

If you want to get the value of a header with an uppercase name, use c.req.header(“X-Foo”).


// ❌ Will not work
const headerRecord = c.req.header()
const foo = headerRecord['X-Foo']

// ✅ Will work
const foo = c.req.header('X-Foo')
parseBody()
Parse Request body of type multipart/form-data or application/x-www-form-urlencoded


app.post('/entry', async (c) => {
  const body = await c.req.parseBody()
  // ...
})
parseBody() supports the following behaviors.

Single file


const body = await c.req.parseBody()
const data = body['foo']
body['foo'] is (string | File).

If multiple files are uploaded, the last one will be used.

Multiple files

const body = await c.req.parseBody()
body['foo[]']
body['foo[]'] is always (string | File)[].

[] postfix is required.

Multiple files or fields with same name
If you have a input field that allows multiple <input type="file" multiple /> or multiple checkboxes with the same name <input type="checkbox" name="favorites" value="Hono"/>.


const body = await c.req.parseBody({ all: true })
body['foo']
all option is disabled by default.

If body['foo'] is multiple files, it will be parsed to (string | File)[].
If body['foo'] is single file, it will be parsed to (string | File).
Dot notation
If you set the dot option true, the return value is structured based on the dot notation.

Imagine receiving the following data:


const data = new FormData()
data.append('obj.key1', 'value1')
data.append('obj.key2', 'value2')
You can get the structured value by setting the dot option true:


const body = await c.req.parseBody({ dot: true })
// body is `{ obj: { key1: 'value1', key2: 'value2' } }`
json()
Parses the request body of type application/json


app.post('/entry', async (c) => {
  const body = await c.req.json()
  // ...
})
text()
Parses the request body of type text/plain


app.post('/entry', async (c) => {
  const body = await c.req.text()
  // ...
})
arrayBuffer()
Parses the request body as an ArrayBuffer


app.post('/entry', async (c) => {
  const body = await c.req.arrayBuffer()
  // ...
})
blob()
Parses the request body as a Blob.


app.post('/entry', async (c) => {
  const body = await c.req.blob()
  // ...
})
formData()
Parses the request body as a FormData.


app.post('/entry', async (c) => {
  const body = await c.req.formData()
  // ...
})
valid()
Get the validated data.


app.post('/posts', async (c) => {
  const { title, body } = c.req.valid('form')
  // ...
})
Available targets are below.

form
json
query
header
cookie
param
See the Validation section for usage examples.

routePath
WARNING

Deprecated in v4.8.0: This property is deprecated. Use routePath() from Route Helper instead.

You can retrieve the registered path within the handler like this:


app.get('/posts/:id', (c) => {
  return c.json({ path: c.req.routePath })
})
If you access /posts/123, it will return /posts/:id:


{ "path": "/posts/:id" }
matchedRoutes
WARNING

Deprecated in v4.8.0: This property is deprecated. Use matchedRoutes() from Route Helper instead.

It returns matched routes within the handler, which is useful for debugging.


app.use(async function logger(c, next) {
  await next()
  c.req.matchedRoutes.forEach(({ handler, method, path }, i) => {
    const name =
      handler.name ||
      (handler.length < 2 ? '[handler]' : '[middleware]')
    console.log(
      method,
      ' ',
      path,
      ' '.repeat(Math.max(10 - path.length, 0)),
      name,
      i === c.req.routeIndex ? '<- respond from here' : ''
    )
  })
})
path
The request pathname.


app.get('/about/me', async (c) => {
  const pathname = c.req.path // `/about/me`
  // ...
})
url
The request url strings.


app.get('/about/me', async (c) => {
  const url = c.req.url // `http://localhost:8787/about/me`
  // ...
})
method
The method name of the request.


app.get('/about/me', async (c) => {
  const method = c.req.method // `GET`
  // ...
})
raw
The raw Request object.


// For Cloudflare Workers
app.post('/', async (c) => {
  const metadata = c.req.raw.cf?.hostMetadata?
  // ...
})
cloneRawRequest()
Clones the raw Request object from a HonoRequest. Works even after the request body has been consumed by validators or HonoRequest methods.


import { Hono } from 'hono'
const app = new Hono()

import { cloneRawRequest } from 'hono/request'
import { validator } from 'hono/validator'

app.post(
  '/forward',
  validator('json', (data) => data),
  async (c) => {
    // Clone after validation
    const clonedReq = await cloneRawRequest(c.req)
    // Does not throw the error
    await clonedReq.json()
    // ...
  }
)
HTTPException
When a fatal error occurs, Hono (and many ecosystem middleware) may throw an HTTPException. This is a custom Hono Error that simplifies returning error responses.

Throwing HTTPExceptions
You can throw your own HTTPExceptions by specifying a status code, and either a message or a custom response.

Custom Message
For basic text responses, just set a the error message.


import { HTTPException } from 'hono/http-exception'

throw new HTTPException(401, { message: 'Unauthorized' })
Custom Response
For other response types, or to set response headers, use the res option. Note that the status passed to the constructor is the one used to create responses.


import { HTTPException } from 'hono/http-exception'

const errorResponse = new Response('Unauthorized', {
  status: 401, // this gets ignored
  headers: {
    Authenticate: 'error="invalid_token"',
  },
})

throw new HTTPException(401, { res: errorResponse })
Cause
In either case, you can use the cause option to add arbitrary data to the HTTPException.


app.post('/login', async (c) => {
  try {
    await authorize(c)
  } catch (cause) {
    throw new HTTPException(401, { message, cause })
  }
  return c.redirect('/')
})
Handling HTTPExceptions
You can handle uncaught HTTPExceptions with app.onError. They include a getResponse method that returns a new Response created from the error status, and either the error message, or the custom response set when the error was thrown.


import { HTTPException } from 'hono/http-exception'

// ...

app.onError((error, c) => {
  if (error instanceof HTTPException) {
    console.error(error.cause)
    // Get the custom response
    return error.getResponse()
  }
  // ...
})
WARNINGPresets
Hono has several routers, each designed for a specific purpose. You can specify the router you want to use in the constructor of Hono.

Presets are provided for common use cases, so you don't have to specify the router each time. The Hono class imported from all presets is the same, the only difference being the router. Therefore, you can use them interchangeably.

hono
Usage:


import { Hono } from 'hono'
Routers:


this.router = new SmartRouter({
  routers: [new RegExpRouter(), new TrieRouter()],
})
hono/quick
Usage:


import { Hono } from 'hono/quick'
Router:


this.router = new SmartRouter({
  routers: [new LinearRouter(), new TrieRouter()],
})
hono/tiny
Usage:


import { Hono } from 'hono/tiny'
Router:


this.router = new PatternRouter()
Which preset should I use?
Preset	Suitable platforms
hono	This is highly recommended for most use cases. Although the registration phase may be slower than hono/quick, it exhibits high performance once booted. It's ideal for long-life servers built with Deno, Bun, or Node.js. It is also suitable for Fastly Compute, as route registration occurs during the app build phase on that platform. For environments such as Cloudflare Workers, Deno Deploy, where v8 isolates are utilized, this preset is suitable as well. Because the isolations persist for a certain amount of time after booting.
hono/quick	This preset is designed for environments where the application is initialized for every request.
hono/tiny	This is the smallest router package and it's suitable for environments where resources are limited.
Create-hono
Command-line options supported by create-hono - the project initializer that runs when you run npm create hono@latest, npx create-hono@latest, or pnpm create hono@latest.

NOTE

Why this page? The installation / quick-start examples often show a minimal npm create hono@latest my-app command. create-hono supports several useful flags you can pass to automate and customize project creation (select templates, skip prompts, pick a package manager, use local cache, and more).

Passing arguments:
When you use npm create (or npx) arguments intended for the initializer script must be placed after --. Anything after -- is forwarded to the initializer.


npm

yarn

pnpm

bun

deno

# Forwarding arguments to create-hono (npm requires `--`)
npm create hono@latest my-app -- --template cloudflare-workers
Commonly used arguments
Argument	Description	Example
--template <template>	Select a starter template and skip the interactive template prompt. Templates may include names like bun, cloudflare-workers, vercel, etc.	--template cloudflare-workers
--install	Automatically install dependencies after the template is created.	--install
--pm <packageManager>	Specify which package manager to run when installing dependencies. Common values: npm, pnpm, yarn.	--pm pnpm
--offline	Use the local cache/templates instead of fetching the latest remote templates. Useful for offline environments or deterministic local runs.	--offline
NOTE

The exact set of templates and available options is maintained by the create-hono project. This docs page summarizes the most-used flags — see the linked repository below for the full, authoritative reference.

Example flows
Minimal, interactive

npm create hono@latest my-app
This prompts you for template and options.

Non-interactive, pick template and package manager

npm create hono@latest my-app -- --template vercel --pm npm --install
This creates my-app using the vercel template, installs dependencies using npm, and skips the interactive prompts.

Use offline cache (no network)

pnpm create hono@latest my-app --template deno --offline
Troubleshooting & tips
If an option appears not to be recognized, make sure you're forwarding it with -- when using npm create / npx .
To see the most current list of templates and flags, consult the create-hono repository or run the initializer locally and follow its help output.
Links & references
create-hono repository : create-hono
Middleware
Middleware works before/after the endpoint Handler. We can get the Request before dispatching or manipulate the Response after dispatching.

Definition of Middleware
Handler - should return Response object. Only one handler will be called.
Middleware - should await next() and return nothing to call the next Middleware, or return a Response to early-exit.
The user can register middleware using app.use or using app.HTTP_METHOD as well as the handlers. For this feature, it's easy to specify the path and the method.


// match any method, all routes
app.use(logger())

// specify path
app.use('/posts/*', cors())

// specify method and path
app.post('/posts/*', basicAuth())
If the handler returns Response, it will be used for the end-user, and stopping the processing.


app.post('/posts', (c) => c.text('Created!', 201))
In this case, four middleware are processed before dispatching like this:


logger() -> cors() -> basicAuth() -> *handler*
Execution order
The order in which Middleware is executed is determined by the order in which it is registered. The process before the next of the first registered Middleware is executed first, and the process after the next is executed last. See below.


app.use(async (_, next) => {
  console.log('middleware 1 start')
  await next()
  console.log('middleware 1 end')
})
app.use(async (_, next) => {
  console.log('middleware 2 start')
  await next()
  console.log('middleware 2 end')
})
app.use(async (_, next) => {
  console.log('middleware 3 start')
  await next()
  console.log('middleware 3 end')
})

app.get('/', (c) => {
  console.log('handler')
  return c.text('Hello!')
})
Result is the following.


middleware 1 start
  middleware 2 start
    middleware 3 start
      handler
    middleware 3 end
  middleware 2 end
middleware 1 end
Note that if the handler or any middleware throws, hono will catch it and either pass it to your app.onError() callback or automatically convert it to a 500 response before returning it up the chain of middleware. This means that next() will never throw, so there is no need to wrap it in a try/catch/finally.

Built-in Middleware
Hono has built-in middleware.


import { Hono } from 'hono'
import { poweredBy } from 'hono/powered-by'
import { logger } from 'hono/logger'
import { basicAuth } from 'hono/basic-auth'

const app = new Hono()

app.use(poweredBy())
app.use(logger())

app.use(
  '/auth/*',
  basicAuth({
    username: 'hono',
    password: 'acoolproject',
  })
)
WARNING

In Deno, it is possible to use a different version of middleware than the Hono version, but this can lead to bugs. For example, this code is not working because the version is different.


import { Hono } from 'jsr:@hono/hono@4.4.0'
import { upgradeWebSocket } from 'jsr:@hono/hono@4.4.5/deno'

const app = new Hono()

app.get(
  '/ws',
  upgradeWebSocket(() => ({
    // ...
  }))
)
Custom Middleware
You can write your own middleware directly inside app.use():


// Custom logger
app.use(async (c, next) => {
  console.log(`[${c.req.method}] ${c.req.url}`)
  await next()
})

// Add a custom header
app.use('/message/*', async (c, next) => {
  await next()
  c.header('x-message', 'This is middleware!')
})

app.get('/message/hello', (c) => c.text('Hello Middleware!'))
However, embedding middleware directly within app.use() can limit its reusability. Therefore, we can separate our middleware into different files.

To ensure we don't lose type definitions for context and next, when separating middleware, we can use createMiddleware() from Hono's factory. This also allows us to type-safely access data we've set in Context from downstream handlers.


import { createMiddleware } from 'hono/factory'

const logger = createMiddleware(async (c, next) => {
  console.log(`[${c.req.method}] ${c.req.url}`)
  await next()
})
INFO

Type generics can be used with createMiddleware:


createMiddleware<{Bindings: Bindings}>(async (c, next) =>
Modify the Response After Next
Additionally, middleware can be designed to modify responses if necessary:


const stripRes = createMiddleware(async (c, next) => {
  await next()
  c.res = undefined
  c.res = new Response('New Response')
})
Context access inside Middleware arguments
To access the context inside middleware arguments, directly use the context parameter provided by app.use. See the example below for clarification.


import { cors } from 'hono/cors'

app.use('*', async (c, next) => {
  const middleware = cors({
    origin: c.env.CORS_ORIGIN,
  })
  return middleware(c, next)
})
Extending the Context in Middleware
To extend the context inside middleware, use c.set. You can make this type-safe by passing a { Variables: { yourVariable: YourVariableType } } generic argument to the createMiddleware function.


import { createMiddleware } from 'hono/factory'

const echoMiddleware = createMiddleware<{
  Variables: {
    echo: (str: string) => string
  }
}>(async (c, next) => {
  c.set('echo', (str) => str)
  await next()
})

app.get('/echo', echoMiddleware, (c) => {
  return c.text(c.var.echo('Hello!'))
})
Third-party Middleware
Built-in middleware does not depend on external modules, but third-party middleware can depend on third-party libraries. So with them, we may make a more complex application.

We can explore a variety of third-party middleware. For example, we have GraphQL Server Middleware, Sentry Middleware, Firebase Auth Middleware, and others.Helpers
Helpers are available to assist in developing your application. Unlike middleware, they don't act as handlers, but rather provide useful functions.

For instance, here's how to use the Cookie helper:


import { getCookie, setCookie } from 'hono/cookie'

const app = new Hono()

app.get('/cookie', (c) => {
  const yummyCookie = getCookie(c, 'yummy_cookie')
  // ...
  setCookie(c, 'delicious_cookie', 'macha')
  //
})
Available Helpers
Accepts
Adapter
Cookie
css
Dev
Factory
html
JWT
SSG
Streaming
Testing
WebSocket
Edit this page on GitHub
Validation
Hono provides only a very thin Validator. But, it can be powerful when combined with a third-party Validator. In addition, the RPC feature allows you to share API specifications with your clients through types.

Manual validator
First, introduce a way to validate incoming values without using the third-party Validator.

Import validator from hono/validator.


import { validator } from 'hono/validator'
To validate form data, specify form as the first argument and a callback as the second argument. In the callback, validates the value and return the validated values at the end. The validator can be used as middleware.


app.post(
  '/posts',
  validator('form', (value, c) => {
    const body = value['body']
    if (!body || typeof body !== 'string') {
      return c.text('Invalid!', 400)
    }
    return {
      body: body,
    }
  }),
  //...
Within the handler you can get the validated value with c.req.valid('form').


, (c) => {
  const { body } = c.req.valid('form')
  // ... do something
  return c.json(
    {
      message: 'Created!',
    },
    201
  )
}
Validation targets include json, query, header, param and cookie in addition to form.

WARNING

When you validate json or form, the request must contain a matching content-type header (e.g. Content-Type: application/json for json). Otherwise, the request body will not be parsed and you will receive an empty object ({}) as value in the callback.

It is important to set the content-type header when testing using app.request().

Given an application like this.


const app = new Hono()
app.post(
  '/testing',
  validator('json', (value, c) => {
    // pass-through validator
    return value
  }),
  (c) => {
    const body = c.req.valid('json')
    return c.json(body)
  }
)
Your tests can be written like this.


// ❌ this will not work
const res = await app.request('/testing', {
  method: 'POST',
  body: JSON.stringify({ key: 'value' }),
})
const data = await res.json()
console.log(data) // {}

// ✅ this will work
const res = await app.request('/testing', {
  method: 'POST',
  body: JSON.stringify({ key: 'value' }),
  headers: new Headers({ 'Content-Type': 'application/json' }),
})
const data = await res.json()
console.log(data) // { key: 'value' }
WARNING

When you validate header, you need to use lowercase name as the key.

If you want to validate the Idempotency-Key header, you need to use idempotency-key as the key.


// ❌ this will not work
app.post(
  '/api',
  validator('header', (value, c) => {
    // idempotencyKey is always undefined
    // so this middleware always return 400 as not expected
    const idempotencyKey = value['Idempotency-Key']

    if (idempotencyKey == undefined || idempotencyKey === '') {
      throw new HTTPException(400, {
        message: 'Idempotency-Key is required',
      })
    }
    return { idempotencyKey }
  }),
  (c) => {
    const { idempotencyKey } = c.req.valid('header')
    // ...
  }
)

// ✅ this will work
app.post(
  '/api',
  validator('header', (value, c) => {
    // can retrieve the value of the header as expected
    const idempotencyKey = value['idempotency-key']

    if (idempotencyKey == undefined || idempotencyKey === '') {
      throw new HTTPException(400, {
        message: 'Idempotency-Key is required',
      })
    }
    return { idempotencyKey }
  }),
  (c) => {
    const { idempotencyKey } = c.req.valid('header')
    // ...
  }
)
Multiple validators
You can also include multiple validators to validate different parts of request:


app.post(
  '/posts/:id',
  validator('param', ...),
  validator('query', ...),
  validator('json', ...),
  (c) => {
    //...
  }
With Zod
You can use Zod, one of third-party validators. We recommend using a third-party validator.

Install from the Npm registry.


npm

yarn

pnpm

bun

npm i zod
Import z from zod.


import * as z from 'zod'
Write your schema.


const schema = z.object({
  body: z.string(),
})
You can use the schema in the callback function for validation and return the validated value.


const route = app.post(
  '/posts',
  validator('form', (value, c) => {
    const parsed = schema.safeParse(value)
    if (!parsed.success) {
      return c.text('Invalid!', 401)
    }
    return parsed.data
  }),
  (c) => {
    const { body } = c.req.valid('form')
    // ... do something
    return c.json(
      {
        message: 'Created!',
      },
      201
    )
  }
)
Zod Validator Middleware
You can use the Zod Validator Middleware to make it even easier.


npm

yarn

pnpm

bun

npm i @hono/zod-validator
And import zValidator.


import { zValidator } from '@hono/zod-validator'
And write as follows.


const route = app.post(
  '/posts',
  zValidator(
    'form',
    z.object({
      body: z.string(),
    })
  ),
  (c) => {
    const validated = c.req.valid('form')
    // ... use your validated data
  }
)
Standard Schema Validator Middleware
Standard Schema is a specification that provides a common interface for TypeScript validation libraries. It was created by the maintainers of Zod, Valibot, and ArkType to allow ecosystem tools to work with any validation library without needing custom adapters.

The Standard Schema Validator Middleware lets you use any Standard Schema-compatible validation library with Hono, giving you the flexibility to choose your preferred validator while maintaining consistent type safety.


npm

yarn

pnpm

bun

npm i @hono/standard-validator
Import sValidator from the package:


import { sValidator } from '@hono/standard-validator'
With Zod
You can use Zod with the Standard Schema validator:


npm

yarn

pnpm

bun

npm i zod

import * as z from 'zod'
import { sValidator } from '@hono/standard-validator'

const schema = z.object({
  name: z.string(),
  age: z.number(),
})

app.post('/author', sValidator('json', schema), (c) => {
  const data = c.req.valid('json')
  return c.json({
    success: true,
    message: `${data.name} is ${data.age}`,
  })
})
With Valibot
Valibot is a lightweight alternative to Zod with a modular design:


npm

yarn

pnpm

bun

npm i valibot

import * as v from 'valibot'
import { sValidator } from '@hono/standard-validator'

const schema = v.object({
  name: v.string(),
  age: v.number(),
})

app.post('/author', sValidator('json', schema), (c) => {
  const data = c.req.valid('json')
  return c.json({
    success: true,
    message: `${data.name} is ${data.age}`,
  })
})
With ArkType
ArkType offers TypeScript-native syntax for runtime validation:


npm

yarn

pnpm

bun

npm i arktype

import { type } from 'arktype'
import { sValidator } from '@hono/standard-validator'

const schema = type({
  name: 'string',
  age: 'number',
})

app.post('/author', sValidator('json', schema), (c) => {
  const data = c.req.valid('json')
  return c.json({
    success: true,
    message: `${data.name} is ${data.age}`,
  })
})
RPC
The RPC feature allows sharing of the API specifications between the server and the client.

First, export the typeof your Hono app (commonly called AppType)—or just the routes you want available to the client—from your server code.

By accepting AppType as a generic parameter, the Hono Client can infer both the input type(s) specified by the Validator, and the output type(s) emitted by handlers returning c.json().

NOTE

For the RPC types to work properly in a monorepo, in both the Client's and Server's tsconfig.json files, set "strict": true in compilerOptions. Read more.

Server
All you need to do on the server side is to write a validator and create a variable route. The following example uses Zod Validator.


const route = app.post(
  '/posts',
  zValidator(
    'form',
    z.object({
      title: z.string(),
      body: z.string(),
    })
  ),
  (c) => {
    // ...
    return c.json(
      {
        ok: true,
        message: 'Created!',
      },
      201
    )
  }
)
Then, export the type to share the API spec with the Client.


export type AppType = typeof route
Client
On the Client side, import hc and AppType first.


import type { AppType } from '.'
import { hc } from 'hono/client'
hc is a function to create a client. Pass AppType as Generics and specify the server URL as an argument.


const client = hc<AppType>('http://localhost:8787/')
Call client.{path}.{method} and pass the data you wish to send to the server as an argument.


const res = await client.posts.$post({
  form: {
    title: 'Hello',
    body: 'Hono is a cool project',
  },
})
The res is compatible with the "fetch" Response. You can retrieve data from the server with res.json().


if (res.ok) {
  const data = await res.json()
  console.log(data.message)
}
Cookies
To make the client send cookies with every request, add { 'init': { 'credentials": 'include' } } to the options when you're creating the client.


// client.ts
const client = hc<AppType>('http://localhost:8787/', {
  init: {
    credentials: 'include',
  },
})

// This request will now include any cookies you might have set
const res = await client.posts.$get({
  query: {
    id: '123',
  },
})
Status code
If you explicitly specify the status code, such as 200 or 404, in c.json(). It will be added as a type for passing to the client.


// server.ts
const app = new Hono().get(
  '/posts',
  zValidator(
    'query',
    z.object({
      id: z.string(),
    })
  ),
  async (c) => {
    const { id } = c.req.valid('query')
    const post: Post | undefined = await getPost(id)

    if (post === undefined) {
      return c.json({ error: 'not found' }, 404) // Specify 404
    }

    return c.json({ post }, 200) // Specify 200
  }
)

export type AppType = typeof app
You can get the data by the status code.


// client.ts
const client = hc<AppType>('http://localhost:8787/')

const res = await client.posts.$get({
  query: {
    id: '123',
  },
})

if (res.status === 404) {
  const data: { error: string } = await res.json()
  console.log(data.error)
}

if (res.ok) {
  const data: { post: Post } = await res.json()
  console.log(data.post)
}

// { post: Post } | { error: string }
type ResponseType = InferResponseType<typeof client.posts.$get>

// { post: Post }
type ResponseType200 = InferResponseType<
  typeof client.posts.$get,
  200
>
Not Found
If you want to use a client, you should not use c.notFound() for the Not Found response. The data that the client gets from the server cannot be inferred correctly.


// server.ts
export const routes = new Hono().get(
  '/posts',
  zValidator(
    'query',
    z.object({
      id: z.string(),
    })
  ),
  async (c) => {
    const { id } = c.req.valid('query')
    const post: Post | undefined = await getPost(id)

    if (post === undefined) {
      return c.notFound() // ❌️
    }

    return c.json({ post })
  }
)

// client.ts
import { hc } from 'hono/client'

const client = hc<typeof routes>('/')

const res = await client.posts[':id'].$get({
  param: {
    id: '123',
  },
})

const data = await res.json() // 🙁 data is unknown
Please use c.json() and specify the status code for the Not Found Response.


export const routes = new Hono().get(
  '/posts',
  zValidator(
    'query',
    z.object({
      id: z.string(),
    })
  ),
  async (c) => {
    const { id } = c.req.valid('query')
    const post = await getPost(id)

    if (!post) {
      return c.json({ error: 'not found' }, 404) // Specify 404
    }

    return c.json({ post }, 200) // Specify 200
  }
)
Alternatively, you can use module augmentation to extend NotFoundResponse interface. This allows c.notFound() to return a typed response:


// server.ts
import { Hono, TypedResponse } from 'hono'

declare module 'hono' {
  interface NotFoundResponse
    extends Response,
      TypedResponse<{ error: string }, 404, 'json'> {}
}

const app = new Hono()
  .get('/posts/:id', async (c) => {
    const post = await getPost(c.req.param('id'))
    if (!post) {
      return c.notFound()
    }
    return c.json({ post }, 200)
  })
  .notFound((c) => c.json({ error: 'not found' }, 404))

export type AppType = typeof app
Now the client can correctly infer the 404 response type.

Path parameters
You can also handle routes that include path parameters or query values.


const route = app.get(
  '/posts/:id',
  zValidator(
    'query',
    z.object({
      page: z.coerce.number().optional(), // coerce to convert to number
    })
  ),
  (c) => {
    // ...
    return c.json({
      title: 'Night',
      body: 'Time to sleep',
    })
  }
)
Both path parameters and query values must be passed as string, even if the underlying value is of a different type.

Specify the string you want to include in the path with param, and any query values with query.


const res = await client.posts[':id'].$get({
  param: {
    id: '123',
  },
  query: {
    page: '1', // `string`, converted by the validator to `number`
  },
})
Multiple parameters
Handle routes with multiple parameters.


const route = app.get(
  '/posts/:postId/:authorId',
  zValidator(
    'query',
    z.object({
      page: z.string().optional(),
    })
  ),
  (c) => {
    // ...
    return c.json({
      title: 'Night',
      body: 'Time to sleep',
    })
  }
)
Add multiple [''] to specify params in path.


const res = await client.posts[':postId'][':authorId'].$get({
  param: {
    postId: '123',
    authorId: '456',
  },
  query: {},
})
Include slashes
hc function does not URL-encode the values of param. To include slashes in parameters, use regular expressions.


// client.ts

// Requests /posts/123/456
const res = await client.posts[':id'].$get({
  param: {
    id: '123/456',
  },
})

// server.ts
const route = app.get(
  '/posts/:id{.+}',
  zValidator(
    'param',
    z.object({
      id: z.string(),
    })
  ),
  (c) => {
    // id: 123/456
    const { id } = c.req.valid('param')
    // ...
  }
)
NOTE

Basic path parameters without regular expressions do not match slashes. If you pass a param containing slashes using the hc function, the server might not route as intended. Encoding the parameters using encodeURIComponent is the recommended approach to ensure correct routing.

Headers
You can append the headers to the request.


const res = await client.search.$get(
  {
    //...
  },
  {
    headers: {
      'X-Custom-Header': 'Here is Hono Client',
      'X-User-Agent': 'hc',
    },
  }
)
To add a common header to all requests, specify it as an argument to the hc function.


const client = hc<AppType>('/api', {
  headers: {
    Authorization: 'Bearer TOKEN',
  },
})
init option
You can pass the fetch's RequestInit object to the request as the init option. Below is an example of aborting a Request.


import { hc } from 'hono/client'

const client = hc<AppType>('http://localhost:8787/')

const abortController = new AbortController()
const res = await client.api.posts.$post(
  {
    json: {
      // Request body
    },
  },
  {
    // RequestInit object
    init: {
      signal: abortController.signal,
    },
  }
)

// ...

abortController.abort()
INFO

A RequestInit object defined by init takes the highest priority. It could be used to overwrite things set by other options like body | method | headers.

$url()
You can get a URL object for accessing the endpoint by using $url().

WARNING

You have to pass in an absolute URL for this to work. Passing in a relative URL / will result in the following error.

Uncaught TypeError: Failed to construct 'URL': Invalid URL


// ❌ Will throw error
const client = hc<AppType>('/')
client.api.post.$url()

// ✅ Will work as expected
const client = hc<AppType>('http://localhost:8787/')
client.api.post.$url()

const route = app
  .get('/api/posts', (c) => c.json({ posts }))
  .get('/api/posts/:id', (c) => c.json({ post }))

const client = hc<typeof route>('http://localhost:8787/')

let url = client.api.posts.$url()
console.log(url.pathname) // `/api/posts`

url = client.api.posts[':id'].$url({
  param: {
    id: '123',
  },
})
console.log(url.pathname) // `/api/posts/123`
Typed URL
You can pass the base URL as the second type parameter to hc to get more precise URL types:


const client = hc<typeof route, 'http://localhost:8787'>(
  'http://localhost:8787/'
)

const url = client.api.posts.$url()
// url is TypedURL with precise type information
// including protocol, host, and path
This is useful when you want to use the URL as a type-safe key for libraries like SWR.

File Uploads
You can upload files using a form body:


// client
const res = await client.user.picture.$put({
  form: {
    file: new File([fileToUpload], filename, {
      type: fileToUpload.type,
    }),
  },
})

// server
const route = app.put(
  '/user/picture',
  zValidator(
    'form',
    z.object({
      file: z.instanceof(File),
    })
  )
  // ...
)
Custom fetch method
You can set the custom fetch method.

In the following example script for Cloudflare Worker, the Service Bindings' fetch method is used instead of the default fetch.


# wrangler.toml
services = [
  { binding = "AUTH", service = "auth-service" },
]

// src/client.ts
const client = hc<CreateProfileType>('http://localhost', {
  fetch: c.env.AUTH.fetch.bind(c.env.AUTH),
})
Custom query serializer
You can customize how query parameters are serialized using the buildSearchParams option. This is useful when you need bracket notation for arrays or other custom formats:


const client = hc<AppType>('http://localhost', {
  buildSearchParams: (query) => {
    const searchParams = new URLSearchParams()
    for (const [k, v] of Object.entries(query)) {
      if (v === undefined) {
        continue
      }
      if (Array.isArray(v)) {
        v.forEach((item) => searchParams.append(`${k}[]`, item))
      } else {
        searchParams.set(k, v)
      }
    }
    return searchParams
  },
})
Infer
Use InferRequestType and InferResponseType to know the type of object to be requested and the type of object to be returned.


import type { InferRequestType, InferResponseType } from 'hono/client'

// InferRequestType
const $post = client.todo.$post
type ReqType = InferRequestType<typeof $post>['form']

// InferResponseType
type ResType = InferResponseType<typeof $post>
Parsing a Response with type-safety helper
You can use parseResponse() helper to easily parse a Response from hc with type-safety.


import { parseResponse, DetailedError } from 'hono/client'

// result contains the parsed response body (automatically parsed based on Content-Type)
const result = await parseResponse(client.hello.$get()).catch(
  (e: DetailedError) => {
    console.error(e)
  }
)
// parseResponse automatically throws an error if response is not ok
Using SWR
You can also use a React Hook library such as SWR.


import useSWR from 'swr'
import { hc } from 'hono/client'
import type { InferRequestType } from 'hono/client'
import type { AppType } from '../functions/api/[[route]]'

const App = () => {
  const client = hc<AppType>('/api')
  const $get = client.hello.$get

  const fetcher =
    (arg: InferRequestType<typeof $get>) => async () => {
      const res = await $get(arg)
      return await res.json()
    }

  const { data, error, isLoading } = useSWR(
    'api-hello',
    fetcher({
      query: {
        name: 'SWR',
      },
    })
  )

  if (error) return <div>failed to load</div>
  if (isLoading) return <div>loading...</div>

  return <h1>{data?.message}</h1>
}

export default App
Using RPC with larger applications
In the case of a larger application, such as the example mentioned in Building a larger application, you need to be careful about the type of inference. A simple way to do this is to chain the handlers so that the types are always inferred.


// authors.ts
import { Hono } from 'hono'

const app = new Hono()
  .get('/', (c) => c.json('list authors'))
  .post('/', (c) => c.json('create an author', 201))
  .get('/:id', (c) => c.json(`get ${c.req.param('id')}`))

export default app

// books.ts
import { Hono } from 'hono'

const app = new Hono()
  .get('/', (c) => c.json('list books'))
  .post('/', (c) => c.json('create a book', 201))
  .get('/:id', (c) => c.json(`get ${c.req.param('id')}`))

export default app
You can then import the sub-routers as you usually would, and make sure you chain their handlers as well, since this is the top level of the app in this case, this is the type we'll want to export.


// index.ts
import { Hono } from 'hono'
import authors from './authors'
import books from './books'

const app = new Hono()

const routes = app.route('/authors', authors).route('/books', books)

export default app
export type AppType = typeof routes
You can now create a new client using the registered AppType and use it as you would normally.

Known issues
IDE performance
When using RPC, the more routes you have, the slower your IDE will become. One of the main reasons for this is that massive amounts of type instantiations are executed to infer the type of your app.

For example, suppose your app has a route like this:


// app.ts
export const app = new Hono().get('foo/:id', (c) =>
  c.json({ ok: true }, 200)
)
Hono will infer the type as follows:


export const app = Hono<BlankEnv, BlankSchema, '/'>().get<
  'foo/:id',
  'foo/:id',
  JSONRespondReturn<{ ok: boolean }, 200>,
  BlankInput,
  BlankEnv
>('foo/:id', (c) => c.json({ ok: true }, 200))
This is a type instantiation for a single route. While the user doesn't need to write these type arguments manually, which is a good thing, it's known that type instantiation takes much time. tsserver used in your IDE does this time consuming task every time you use the app. If you have a lot of routes, this can slow down your IDE significantly.

However, we have some tips to mitigate this issue.

Hono version mismatch
If your backend is separate from the frontend and lives in a different directory, you need to ensure that the Hono versions match. If you use one Hono version on the backend and another on the frontend, you'll run into issues such as "Type instantiation is excessively deep and possibly infinite".



TypeScript project references
Like in the case of Hono version mismatch, you'll run into issues if your backend and frontend are separate. If you want to access code from the backend (AppType, for example) on the frontend, you need to use project references. TypeScript's project references allow one TypeScript codebase to access and use code from another TypeScript codebase. (source: Hono RPC And TypeScript Project References).

Compile your code before using it (recommended)
tsc can do heavy tasks like type instantiation at compile time! Then, tsserver doesn't need to instantiate all the type arguments every time you use it. It will make your IDE a lot faster!

Compiling your client including the server app gives you the best performance. Put the following code in your project:


import { app } from './app'
import { hc } from 'hono/client'

// this is a trick to calculate the type when compiling
export type Client = ReturnType<typeof hc<typeof app>>

export const hcWithType = (...args: Parameters<typeof hc>): Client =>
  hc<typeof app>(...args)
After compiling, you can use hcWithType instead of hc to get the client with the type already calculated.


const client = hcWithType('http://localhost:8787/')
const res = await client.posts.$post({
  form: {
    title: 'Hello',
    body: 'Hono is a cool project',
  },
})
If your project is a monorepo, this solution does fit well. Using a tool like turborepo, you can easily separate the server project and the client project and get better integration managing dependencies between them. Here is a working example.

You can also coordinate your build process manually with tools like concurrently or npm-run-all.

Specify type arguments manually
This is a bit cumbersome, but you can specify type arguments manually to avoid type instantiation.


const app = new Hono().get<'foo/:id'>('foo/:id', (c) =>
  c.json({ ok: true }, 200)
)
Specifying just single type argument make a difference in performance, while it may take you a lot of time and effort if you have a lot of routes.

Split your app and client into multiple files
As described in Using RPC with larger applications, you can split your app into multiple apps. You can also create a client for each app:


// authors-cli.ts
import { app as authorsApp } from './authors'
import { hc } from 'hono/client'

const authorsClient = hc<typeof authorsApp>('/authors')

// books-cli.ts
import { app as booksApp } from './books'
import { hc } from 'hono/client'

const booksClient = hc<typeof booksApp>('/books')
This way, tsserver doesn't need to instantiate types for all routes at once.

Edit this page on GitHubBest Practices
Hono is very flexible. You can write your app as you like. However, there are best practices that are better to follow.

Don't make "Controllers" when possible
When possible, you should not create "Ruby on Rails-like Controllers".


// 🙁
// A RoR-like Controller
const booksList = (c: Context) => {
  return c.json('list books')
}

app.get('/books', booksList)
The issue is related to types. For example, the path parameter cannot be inferred in the Controller without writing complex generics.


// 🙁
// A RoR-like Controller
const bookPermalink = (c: Context) => {
  const id = c.req.param('id') // Can't infer the path param
  return c.json(`get ${id}`)
}
Therefore, you don't need to create RoR-like controllers and should write handlers directly after path definitions.


// 😃
app.get('/books/:id', (c) => {
  const id = c.req.param('id') // Can infer the path param
  return c.json(`get ${id}`)
})
factory.createHandlers() in hono/factory
If you still want to create a RoR-like Controller, use factory.createHandlers() in hono/factory. If you use this, type inference will work correctly.


import { createFactory } from 'hono/factory'
import { logger } from 'hono/logger'

// ...

// 😃
const factory = createFactory()

const middleware = factory.createMiddleware(async (c, next) => {
  c.set('foo', 'bar')
  await next()
})

const handlers = factory.createHandlers(logger(), middleware, (c) => {
  return c.json(c.var.foo)
})

app.get('/api', ...handlers)
Building a larger application
Use app.route() to build a larger application without creating "Ruby on Rails-like Controllers".

If your application has /authors and /books endpoints and you wish to separate files from index.ts, create authors.ts and books.ts.


// authors.ts
import { Hono } from 'hono'

const app = new Hono()

app.get('/', (c) => c.json('list authors'))
app.post('/', (c) => c.json('create an author', 201))
app.get('/:id', (c) => c.json(`get ${c.req.param('id')}`))

export default app

// books.ts
import { Hono } from 'hono'

const app = new Hono()

app.get('/', (c) => c.json('list books'))
app.post('/', (c) => c.json('create a book', 201))
app.get('/:id', (c) => c.json(`get ${c.req.param('id')}`))

export default app
Then, import them and mount on the paths /authors and /books with app.route().


// index.ts
import { Hono } from 'hono'
import authors from './authors'
import books from './books'

const app = new Hono()

// 😃
app.route('/authors', authors)
app.route('/books', books)

export default app
If you want to use RPC features
The code above works well for normal use cases. However, if you want to use the RPC feature, you can get the correct type by chaining as follows.


// authors.ts
import { Hono } from 'hono'

const app = new Hono()
  .get('/', (c) => c.json('list authors'))
  .post('/', (c) => c.json('create an author', 201))
  .get('/:id', (c) => c.json(`get ${c.req.param('id')}`))

export default app
export type AppType = typeof app
If you pass the type of the app to hc, it will get the correct type.


import type { AppType } from './authors'
import { hc } from 'hono/client'

// 😃
const client = hc<AppType>('http://localhost') // Typed correctly
For more detailed information, please see the RPC page.Helpers
Accepts

Adapter

ConnInfo

Cookie

css

Dev

Factory

html

JWT

Proxy

Route

SSG

Streaming

Testing

WebSocketStreaming Helper
The Streaming Helper provides methods for streaming responses.

Import

import { Hono } from 'hono'
import { stream, streamText, streamSSE } from 'hono/streaming'
stream()
It returns a simple streaming response as Response object.


app.get('/stream', (c) => {
  return stream(c, async (stream) => {
    // Write a process to be executed when aborted.
    stream.onAbort(() => {
      console.log('Aborted!')
    })
    // Write a Uint8Array.
    await stream.write(new Uint8Array([0x48, 0x65, 0x6c, 0x6c, 0x6f]))
    // Pipe a readable stream.
    await stream.pipe(anotherReadableStream)
  })
})
streamText()
It returns a streaming response with Content-Type:text/plain, Transfer-Encoding:chunked, and X-Content-Type-Options:nosniff headers.


app.get('/streamText', (c) => {
  return streamText(c, async (stream) => {
    // Write a text with a new line ('\n').
    await stream.writeln('Hello')
    // Wait 1 second.
    await stream.sleep(1000)
    // Write a text without a new line.
    await stream.write(`Hono!`)
  })
})
WARNING

If you are developing an application for Cloudflare Workers, a streaming may not work well on Wrangler. If so, add Identity for Content-Encoding header.


app.get('/streamText', (c) => {
  c.header('Content-Encoding', 'Identity')
  return streamText(c, async (stream) => {
    // ...
  })
})
streamSSE()
It allows you to stream Server-Sent Events (SSE) seamlessly.


const app = new Hono()
let id = 0

app.get('/sse', async (c) => {
  return streamSSE(c, async (stream) => {
    while (true) {
      const message = `It is ${new Date().toISOString()}`
      await stream.writeSSE({
        data: message,
        event: 'time-update',
        id: String(id++),
      })
      await stream.sleep(1000)
    }
  })
})
Error Handling
The third argument of the streaming helper is an error handler. This argument is optional, if you don't specify it, the error will be output as a console error.


app.get('/stream', (c) => {
  return stream(
    c,
    async (stream) => {
      // Write a process to be executed when aborted.
      stream.onAbort(() => {
        console.log('Aborted!')
      })
      // Write a Uint8Array.
      await stream.write(
        new Uint8Array([0x48, 0x65, 0x6c, 0x6c, 0x6f])
      )
      // Pipe a readable stream.
      await stream.pipe(anotherReadableStream)
    },
    (err, stream) => {
      stream.writeln('An error occurred!')
      console.error(err)
    }
  )
})
The stream will be automatically closed after the callbacks are executed.

WARNING

If the callback function of the streaming helper throws an error, the onError event of Hono will not be triggered.

onError is a hook to handle errors before the response is sent and overwrite the response. However, when the callback function is executed, the stream has already started, so it cannot be overwritten.

Edit this page on GitHub
SSG Helper
SSG Helper generates a static site from your Hono application. It will retrieve the contents of registered routes and save them as static files.

Usage
Manual
If you have a simple Hono application like the following:


// index.tsx
const app = new Hono()

app.get('/', (c) => c.html('Hello, World!'))

app.use('/about', async (c, next) => {
  c.setRenderer((content) => {
    return c.html(
      <html>
        <head />
        <body>
          <p>{content}</p>
        </body>
      </html>
    )
  })
  await next()
})

app.get('/about', (c) => {
  return c.render(
    <>
      <title>Hono SSG Page</title>Hello!
    </>
  )
})

export default app
For Node.js, create a build script like this:


// build.ts
import app from './index'
import { toSSG } from 'hono/ssg'
import fs from 'fs/promises'

toSSG(app, fs)
By executing the script, the files will be output as follows:


ls ./static
about.html  index.html
Vite Plugin
Using the @hono/vite-ssg Vite Plugin, you can easily handle the process.

For more details, see here:

https://github.com/honojs/vite-plugins/tree/main/packages/ssg

toSSG
toSSG is the main function for generating static sites, taking an application and a filesystem module as arguments. It is based on the following:

Input
The arguments for toSSG are specified in ToSSGInterface.


export interface ToSSGInterface {
  (
    app: Hono,
    fsModule: FileSystemModule,
    options?: ToSSGOptions
  ): Promise<ToSSGResult>
}
app specifies new Hono() with registered routes.
fs specifies the following object, assuming node:fs/promise.

export interface FileSystemModule {
  writeFile(path: string, data: string | Uint8Array): Promise<void>
  mkdir(
    path: string,
    options: { recursive: boolean }
  ): Promise<void | string>
}
Using adapters for Deno and Bun
If you want to use SSG on Deno or Bun, a toSSG function is provided for each file system.

For Deno:


import { toSSG } from 'hono/deno'

toSSG(app) // The second argument is an option typed `ToSSGOptions`.
For Bun:


import { toSSG } from 'hono/bun'

toSSG(app) // The second argument is an option typed `ToSSGOptions`.
Options
Options are specified in the ToSSGOptions interface.


export interface ToSSGOptions {
  dir?: string
  concurrency?: number
  extensionMap?: Record<string, string>
  plugins?: SSGPlugin[]
}
dir is the output destination for Static files. The default value is ./static.
concurrency is the concurrent number of files to be generated at the same time. The default value is 2.
extensionMap is a map containing the Content-Type as a key and the string of the extension as a value. This is used to determine the file extension of the output file.
plugins is an array of SSG plugins that extend the functionality of the static site generation process.
Output
toSSG returns the result in the following Result type.


export interface ToSSGResult {
  success: boolean
  files: string[]
  error?: Error
}
Generate File
Route and Filename
The following rules apply to the registered route information and the generated file name. The default ./static behaves as follows:

/ -> ./static/index.html
/path -> ./static/path.html
/path/ -> ./static/path/index.html
File Extension
The file extension depends on the Content-Type returned by each route. For example, responses from c.html are saved as .html.

If you want to customize the file extensions, set the extensionMap option.


import { toSSG, defaultExtensionMap } from 'hono/ssg'

// Save `application/x-html` content with `.html`
toSSG(app, fs, {
  extensionMap: {
    'application/x-html': 'html',
    ...defaultExtensionMap,
  },
})
Note that paths ending with a slash are saved as index.ext regardless of the extension.


// save to ./static/html/index.html
app.get('/html/', (c) => c.html('html'))

// save to ./static/text/index.txt
app.get('/text/', (c) => c.text('text'))
Middleware
Introducing built-in middleware that supports SSG.

ssgParams
You can use an API like generateStaticParams of Next.js.

Example:


app.get(
  '/shops/:id',
  ssgParams(async () => {
    const shops = await getShops()
    return shops.map((shop) => ({ id: shop.id }))
  }),
  async (c) => {
    const shop = await getShop(c.req.param('id'))
    if (!shop) {
      return c.notFound()
    }
    return c.render(
      <div>
        <h1>{shop.name}</h1>
      </div>
    )
  }
)
disableSSG
Routes with the disableSSG middleware set are excluded from static file generation by toSSG.


app.get('/api', disableSSG(), (c) => c.text('an-api'))
onlySSG
Routes with the onlySSG middleware set will be overridden by c.notFound() after toSSG execution.


app.get('/static-page', onlySSG(), (c) => c.html(<h1>Welcome to my site</h1>))
Plugins
Plugins allow you to extend the functionality of the static site generation process. They use hooks to customize the generation process at different stages.

Default Plugin
By default, toSSG uses defaultPlugin which skips non-200 status responses (like redirects, errors, or 404s). This prevents generating files for unsuccessful responses.


import { toSSG, defaultPlugin } from 'hono/ssg'

// defaultPlugin is automatically applied when no plugins specified
toSSG(app, fs)

// Equivalent to:
toSSG(app, fs, { plugins: [defaultPlugin] })
If you specify custom plugins, defaultPlugin is not automatically included. To keep the default behavior while adding custom plugins, explicitly include it:


toSSG(app, fs, {
  plugins: [defaultPlugin, myCustomPlugin],
})
Hook Types
Plugins can use the following hooks to customize the toSSG process:


export type BeforeRequestHook = (req: Request) => Request | false
export type AfterResponseHook = (res: Response) => Response | false
export type AfterGenerateHook = (
  result: ToSSGResult
) => void | Promise<void>
BeforeRequestHook: Called before processing each request. Return false to skip the route.
AfterResponseHook: Called after receiving each response. Return false to skip file generation.
AfterGenerateHook: Called after the entire generation process completes.
Plugin Interface

export interface SSGPlugin {
  beforeRequestHook?: BeforeRequestHook | BeforeRequestHook[]
  afterResponseHook?: AfterResponseHook | AfterResponseHook[]
  afterGenerateHook?: AfterGenerateHook | AfterGenerateHook[]
}
Basic Plugin Examples
Filter only GET requests:


const getOnlyPlugin: SSGPlugin = {
  beforeRequestHook: (req) => {
    if (req.method === 'GET') {
      return req
    }
    return false
  },
}
Filter by status code:


const statusFilterPlugin: SSGPlugin = {
  afterResponseHook: (res) => {
    if (res.status === 200 || res.status === 500) {
      return res
    }
    return false
  },
}
Log generated files:


const logFilesPlugin: SSGPlugin = {
  afterGenerateHook: (result) => {
    if (result.files) {
      result.files.forEach((file) => console.log(file))
    }
  },
}
Advanced Plugin Example
Here's an example of creating a sitemap plugin that generates a sitemap.xml file:


// plugins.ts
import fs from 'node:fs/promises'
import path from 'node:path'
import type { SSGPlugin } from 'hono/ssg'
import { DEFAULT_OUTPUT_DIR } from 'hono/ssg'

export const sitemapPlugin = (baseURL: string): SSGPlugin => {
  return {
    afterGenerateHook: (result, fsModule, options) => {
      const outputDir = options?.dir ?? DEFAULT_OUTPUT_DIR
      const filePath = path.join(outputDir, 'sitemap.xml')
      const urls = result.files.map((file) =>
        new URL(file, baseURL).toString()
      )
      const siteMapText = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls.map((url) => `<url><loc>${url}</loc></url>`).join('\n')}
</urlset>`
      fsModule.writeFile(filePath, siteMapText)
    },
  }
}
Applying plugins:


import app from './index'
import { toSSG } from 'hono/ssg'
import { sitemapPlugin } from './plugins'

toSSG(app, fs, {
  plugins: [
    getOnlyPlugin,
    statusFilterPlugin,
    logFilesPlugin,
    sitemapPlugin('https://example.com'),
  ],
})html Helper
The html Helper lets you write HTML in JavaScript template literal with a tag named html. Using raw(), the content will be rendered as is. You have to escape these strings by yourself.

Import

import { Hono } from 'hono'
import { html, raw } from 'hono/html'
html

const app = new Hono()

app.get('/:username', (c) => {
  const { username } = c.req.param()
  return c.html(
    html`<!doctype html>
      <h1>Hello! ${username}!</h1>`
  )
})
Insert snippets into JSX
Insert the inline script into JSX:


app.get('/', (c) => {
  return c.html(
    <html>
      <head>
        <title>Test Site</title>
        {html`
          <script>
            // No need to use dangerouslySetInnerHTML.
            // If you write it here, it will not be escaped.
          </script>
        `}
      </head>
      <body>Hello!</body>
    </html>
  )
})
Act as functional component
Since html returns an HtmlEscapedString, it can act as a fully functional component without using JSX.

Use html to speed up the process instead of memo

const Footer = () => html`
  <footer>
    <address>My Address...</address>
  </footer>
`
Receives props and embeds values

interface SiteData {
  title: string
  description: string
  image: string
  children?: any
}
const Layout = (props: SiteData) => html`
<html>
<head>
  <meta charset="UTF-8">
  <title>${props.title}</title>
  <meta name="description" content="${props.description}">
  <head prefix="og: http://ogp.me/ns#">
  <meta property="og:type" content="article">
  <!-- More elements slow down JSX, but not template literals. -->
  <meta property="og:title" content="${props.title}">
  <meta property="og:image" content="${props.image}">
</head>
<body>
  ${props.children}
</body>
</html>
`

const Content = (props: { siteData: SiteData; name: string }) => (
  <Layout {...props.siteData}>
    <h1>Hello {props.name}</h1>
  </Layout>
)

app.get('/', (c) => {
  const props = {
    name: 'World',
    siteData: {
      title: 'Hello <> World',
      description: 'This is a description',
      image: 'https://example.com/image.png',
    },
  }
  return c.html(<Content {...props} />)
})
raw()

app.get('/', (c) => {
  const name = 'John &quot;Johnny&quot; Smith'
  return c.html(html`<p>I'm ${raw(name)}.</p>`)
})
Tips
Thanks to these libraries, Visual Studio Code and vim also interprets template literals as HTML, allowing syntax highlighting and formatting to be applied.

https://marketplace.visualstudio.com/items?itemName=bierner.lit-html
https://github.com/MaxMEllon/vim-jsx-pretty
Cookie Helper
The Cookie Helper provides an easy interface to manage cookies, enabling developers to set, parse, and delete cookies seamlessly.

Import

import { Hono } from 'hono'
import {
  deleteCookie,
  getCookie,
  getSignedCookie,
  setCookie,
  setSignedCookie,
  generateCookie,
  generateSignedCookie,
} from 'hono/cookie'
Usage
Regular cookies

app.get('/cookie', (c) => {
  setCookie(c, 'cookie_name', 'cookie_value')
  const yummyCookie = getCookie(c, 'cookie_name')
  deleteCookie(c, 'cookie_name')
  const allCookies = getCookie(c)
  // ...
})
Signed cookies
NOTE: Setting and retrieving signed cookies returns a Promise due to the async nature of the WebCrypto API, which is used to create HMAC SHA-256 signatures.


app.get('/signed-cookie', (c) => {
  const secret = 'secret' // make sure it's a large enough string to be secure

  await setSignedCookie(c, 'cookie_name0', 'cookie_value', secret)
  const fortuneCookie = await getSignedCookie(
    c,
    secret,
    'cookie_name0'
  )
  deleteCookie(c, 'cookie_name0')
  // `getSignedCookie` will return `false` for a specified cookie if the signature was tampered with or is invalid
  const allSignedCookies = await getSignedCookie(c, secret)
  // ...
})
Cookie Generation
generateCookie and generateSignedCookie functions allow you to create cookie strings directly without setting them in the response headers.

generateCookie

// Basic cookie generation
const cookie = generateCookie('delicious_cookie', 'macha')
// Returns: 'delicious_cookie=macha; Path=/'

// Cookie with options
const cookie = generateCookie('delicious_cookie', 'macha', {
  path: '/',
  secure: true,
  httpOnly: true,
  domain: 'example.com',
})
generateSignedCookie

// Basic signed cookie generation
const signedCookie = await generateSignedCookie(
  'delicious_cookie',
  'macha',
  'secret chocolate chips'
)

// Signed cookie with options
const signedCookie = await generateSignedCookie(
  'delicious_cookie',
  'macha',
  'secret chocolate chips',
  {
    path: '/',
    secure: true,
    httpOnly: true,
  }
)
Note: Unlike setCookie and setSignedCookie, these functions only generate the cookie strings. You need to manually set them in headers if needed.

Options
setCookie & setSignedCookie
domain: string
expires: Date
httpOnly: boolean
maxAge: number
path: string
secure: boolean
sameSite: 'Strict' | 'Lax' | 'None'
priority: 'Low' | 'Medium' | 'High'
prefix: secure | 'host'
partitioned: boolean
Example:


// Regular cookies
setCookie(c, 'great_cookie', 'banana', {
  path: '/',
  secure: true,
  domain: 'example.com',
  httpOnly: true,
  maxAge: 1000,
  expires: new Date(Date.UTC(2000, 11, 24, 10, 30, 59, 900)),
  sameSite: 'Strict',
})

// Signed cookies
await setSignedCookie(
  c,
  'fortune_cookie',
  'lots-of-money',
  'secret ingredient',
  {
    path: '/',
    secure: true,
    domain: 'example.com',
    httpOnly: true,
    maxAge: 1000,
    expires: new Date(Date.UTC(2000, 11, 24, 10, 30, 59, 900)),
    sameSite: 'Strict',
  }
)
deleteCookie
path: string
secure: boolean
domain: string
Example:


deleteCookie(c, 'banana', {
  path: '/',
  secure: true,
  domain: 'example.com',
})
deleteCookie returns the deleted value:


const deletedCookie = deleteCookie(c, 'delicious_cookie')
__Secure- and __Host- prefix
The Cookie helper supports __Secure- and __Host- prefix for cookies names.

If you want to verify if the cookie name has a prefix, specify the prefix option.


const securePrefixCookie = getCookie(c, 'yummy_cookie', 'secure')
const hostPrefixCookie = getCookie(c, 'yummy_cookie', 'host')

const securePrefixSignedCookie = await getSignedCookie(
  c,
  secret,
  'fortune_cookie',
  'secure'
)
const hostPrefixSignedCookie = await getSignedCookie(
  c,
  secret,
  'fortune_cookie',
  'host'
)
Also, if you wish to specify a prefix when setting the cookie, specify a value for the prefix option.


setCookie(c, 'delicious_cookie', 'macha', {
  prefix: 'secure', // or `host`
})

await setSignedCookie(
  c,
  'delicious_cookie',
  'macha',
  'secret choco chips',
  {
    prefix: 'secure', // or `host`
  }
)
Following the best practices
A New Cookie RFC (a.k.a cookie-bis) and CHIPS include some best practices for Cookie settings that developers should follow.

RFC6265bis-13
Max-Age/Expires limitation
__Host-/__Secure- prefix limitation
CHIPS-01
Partitioned limitation
Hono is following the best practices. The cookie helper will throw an Error when parsing cookies under the following conditions:

The cookie name starts with __Secure-, but secure option is not set.
The cookie name starts with __Host-, but secure option is not set.
The cookie name starts with __Host-, but path is not /.
The cookie name starts with __Host-, but domain is set.
The maxAge option value is greater than 400 days.
The expires option value is 400 days later than the current time.
Adapter Helper
The Adapter Helper provides a seamless way to interact with various platforms through a unified interface.

Import

import { Hono } from 'hono'
import { env, getRuntimeKey } from 'hono/adapter'
env()
The env() function facilitates retrieving environment variables across different runtimes, extending beyond just Cloudflare Workers' Bindings. The value that can be retrieved with env(c) may be different for each runtimes.


import { env } from 'hono/adapter'

app.get('/env', (c) => {
  // NAME is process.env.NAME on Node.js or Bun
  // NAME is the value written in `wrangler.toml` on Cloudflare
  const { NAME } = env<{ NAME: string }>(c)
  return c.text(NAME)
})
Supported Runtimes, Serverless Platforms and Cloud Services:

Cloudflare Workers
wrangler.toml
wrangler.jsonc
Deno
Deno.env
.env file
Bun
Bun.env
process.env
Node.js
process.env
Vercel
Environment Variables on Vercel
AWS Lambda
Environment Variables on AWS Lambda
Lambda@Edge
Environment Variables on Lambda are not supported by Lambda@Edge, you need to use Lamdba@Edge event as an alternative.
Fastly Compute
On Fastly Compute, you can use the ConfigStore to manage user-defined data.
Netlify
On Netlify, you can use the Netlify Contexts to manage user-defined data.
Specify the runtime
You can specify the runtime to get environment variables by passing the runtime key as the second argument.


app.get('/env', (c) => {
  const { NAME } = env<{ NAME: string }>(c, 'workerd')
  return c.text(NAME)
})
getRuntimeKey()
The getRuntimeKey() function returns the identifier of the current runtime.


app.get('/', (c) => {
  if (getRuntimeKey() === 'workerd') {
    return c.text('You are on Cloudflare')
  } else if (getRuntimeKey() === 'bun') {
    return c.text('You are on Bun')
  }
  ...
})
Available Runtimes Keys
Here are the available runtimes keys, unavailable runtime key runtimes may be supported and labeled as other, with some being inspired by WinterCG's Runtime Keys:

workerd - Cloudflare Workers
deno
bun
node
edge-light - Vercel Edge Functions
fastly - Fastly Compute
other - Other unknown runtimes keys
Edit this page on GitHub

Accepts Helper
Accepts Helper helps to handle Accept headers in the Requests.

Import

import { Hono } from 'hono'
import { accepts } from 'hono/accepts'
accepts()
The accepts() function looks at the Accept header, such as Accept-Encoding and Accept-Language, and returns the proper value.


import { accepts } from 'hono/accepts'

app.get('/', (c) => {
  const accept = accepts(c, {
    header: 'Accept-Language',
    supports: ['en', 'ja', 'zh'],
    default: 'en',
  })
  return c.json({ lang: accept })
})
AcceptHeader type
The definition of the AcceptHeader type is as follows.


export type AcceptHeader =
  | 'Accept'
  | 'Accept-Charset'
  | 'Accept-Encoding'
  | 'Accept-Language'
  | 'Accept-Patch'
  | 'Accept-Post'
  | 'Accept-Ranges'
Options
required header: AcceptHeader
The target accept header.

required supports: string[]
The header values which your application supports.

required default: string
The default values.

optional match: (accepts: Accept[], config: acceptsConfig) => stringWebSocket Helper
WebSocket Helper is a helper for server-side WebSockets in Hono applications. Currently Cloudflare Workers / Pages, Deno, and Bun adapters are available.

Import

Cloudflare Workers

Deno

Bun

import { Hono } from 'hono'
import { upgradeWebSocket } from 'hono/cloudflare-workers'
If you use Node.js, you can use @hono/node-ws.

upgradeWebSocket()
upgradeWebSocket() returns a handler for handling WebSocket.


const app = new Hono()

app.get(
  '/ws',
  upgradeWebSocket((c) => {
    return {
      onMessage(event, ws) {
        console.log(`Message from client: ${event.data}`)
        ws.send('Hello from server!')
      },
      onClose: () => {
        console.log('Connection closed')
      },
    }
  })
)
Available events:

onOpen - Currently, Cloudflare Workers does not support it.
onMessage
onClose
onError
WARNING

If you use middleware that modifies headers (e.g., applying CORS) on a route that uses WebSocket Helper, you may encounter an error saying you can't modify immutable headers. This is because upgradeWebSocket() also changes headers internally.

Therefore, please be cautious if you are using WebSocket Helper and middleware at the same time.

RPC-mode
Handlers defined with WebSocket Helper support RPC mode.


// server.ts
const wsApp = app.get(
  '/ws',
  upgradeWebSocket((c) => {
    //...
  })
)

export type WebSocketApp = typeof wsApp

// client.ts
const client = hc<WebSocketApp>('http://localhost:8787')
const socket = client.ws.$ws() // A WebSocket object for a client
Examples
See the examples using WebSocket Helper.

Server and Client

// server.ts
import { Hono } from 'hono'
import { upgradeWebSocket } from 'hono/cloudflare-workers'

const app = new Hono().get(
  '/ws',
  upgradeWebSocket(() => {
    return {
      onMessage: (event) => {
        console.log(event.data)
      },
    }
  })
)

export default app

// client.ts
import { hc } from 'hono/client'
import type app from './server'

const client = hc<typeof app>('http://localhost:8787')
const ws = client.ws.$ws(0)

ws.addEventListener('open', () => {
  setInterval(() => {
    ws.send(new Date().toString())
  }, 1000)
})
Bun with JSX

import { Hono } from 'hono'
import { upgradeWebSocket, websocket } from 'hono/bun'
import { html } from 'hono/html'

const app = new Hono()

app.get('/', (c) => {
  return c.html(
    <html>
      <head>
        <meta charset='UTF-8' />
      </head>
      <body>
        <div id='now-time'></div>
        {html`
          <script>
            const ws = new WebSocket('ws://localhost:3000/ws')
            const $nowTime = document.getElementById('now-time')
            ws.onmessage = (event) => {
              $nowTime.textContent = event.data
            }
          </script>
        `}
      </body>
    </html>
  )
})

const ws = app.get(
  '/ws',
  upgradeWebSocket((c) => {
    let intervalId
    return {
      onOpen(_event, ws) {
        intervalId = setInterval(() => {
          ws.send(new Date().toString())
        }, 200)
      },
      onClose() {
        clearInterval(intervalId)
      },
    }
  })
)

export default {
  fetch: app.fetch,
  websocket,
}Route Helper
The Route Helper provides enhanced routing information for debugging and middleware development. It allows you to access detailed information about matched routes and the current route being processed.

Import

import { Hono } from 'hono'
import {
  matchedRoutes,
  routePath,
  baseRoutePath,
  basePath,
} from 'hono/route'
Usage
Basic route information

const app = new Hono()

app.get('/posts/:id', (c) => {
  const currentPath = routePath(c) // '/posts/:id'
  const routes = matchedRoutes(c) // Array of matched routes

  return c.json({
    path: currentPath,
    totalRoutes: routes.length,
  })
})
Working with sub-applications

const app = new Hono()
const apiApp = new Hono()

apiApp.get('/posts/:id', (c) => {
  return c.json({
    routePath: routePath(c), // '/posts/:id'
    baseRoutePath: baseRoutePath(c), // '/api'
    basePath: basePath(c), // '/api' (with actual params)
  })
})

app.route('/api', apiApp)
matchedRoutes()
Returns an array of all routes that matched the current request, including middleware.


app.all('/api/*', (c, next) => {
  console.log('API middleware')
  return next()
})

app.get('/api/users/:id', (c) => {
  const routes = matchedRoutes(c)
  // Returns: [
  //   { method: 'ALL', path: '/api/*', handler: [Function] },
  //   { method: 'GET', path: '/api/users/:id', handler: [Function] }
  // ]
  return c.json({ routes: routes.length })
})
routePath()
Returns the route path pattern registered for the current handler.


app.get('/posts/:id', (c) => {
  console.log(routePath(c)) // '/posts/:id'
  return c.text('Post details')
})
Using with index parameter
You can optionally pass an index parameter to get the route path at a specific position, similar to Array.prototype.at().


app.all('/api/*', (c, next) => {
  return next()
})

app.get('/api/users/:id', (c) => {
  console.log(routePath(c, 0)) // '/api/*' (first matched route)
  console.log(routePath(c, -1)) // '/api/users/:id' (last matched route)
  return c.text('User details')
})
baseRoutePath()
Returns the base path pattern of the current route as specified in routing.


const subApp = new Hono()
subApp.get('/posts/:id', (c) => {
  return c.text(baseRoutePath(c)) // '/:sub'
})

app.route('/:sub', subApp)
Using with index parameter
You can optionally pass an index parameter to get the base route path at a specific position, similar to Array.prototype.at().


app.all('/api/*', (c, next) => {
  return next()
})

const subApp = new Hono()
subApp.get('/users/:id', (c) => {
  console.log(baseRoutePath(c, 0)) // '/' (first matched route)
  console.log(baseRoutePath(c, -1)) // '/api' (last matched route)
  return c.text('User details')
})

app.route('/api', subApp)
basePath()
Returns the base path with embedded parameters from the actual request.


const subApp = new Hono()
subApp.get('/posts/:id', (c) => {
  return c.text(basePath(c)) // '/api' (for request to '/api/posts/123')
})

app.route('/:sub', subApp)Bearer Auth Middleware
The Bearer Auth Middleware provides authentication by verifying an API token in the Request header. The HTTP clients accessing the endpoint will add the Authorization header with Bearer {token} as the header value.

Using curl from the terminal, it would look like this:


curl -H 'Authorization: Bearer honoiscool' http://localhost:8787/auth/page
Import

import { Hono } from 'hono'
import { bearerAuth } from 'hono/bearer-auth'
Usage
NOTE

Your token must match the regex /[A-Za-z0-9._~+/-]+=*/, otherwise a 400 error will be returned. Notably, this regex accommodates both URL-safe Base64- and standard Base64-encoded JWTs. This middleware does not require the bearer token to be a JWT, just that it matches the above regex.


const app = new Hono()

const token = 'honoiscool'

app.use('/api/*', bearerAuth({ token }))

app.get('/api/page', (c) => {
  return c.json({ message: 'You are authorized' })
})
To restrict to a specific route + method:


const app = new Hono()

const token = 'honoiscool'

app.get('/api/page', (c) => {
  return c.json({ message: 'Read posts' })
})

app.post('/api/page', bearerAuth({ token }), (c) => {
  return c.json({ message: 'Created post!' }, 201)
})
To implement multiple tokens (E.g., any valid token can read but create/update/delete are restricted to a privileged token):


const app = new Hono()

const readToken = 'read'
const privilegedToken = 'read+write'
const privilegedMethods = ['POST', 'PUT', 'PATCH', 'DELETE']

app.on('GET', '/api/page/*', async (c, next) => {
  // List of valid tokens
  const bearer = bearerAuth({ token: [readToken, privilegedToken] })
  return bearer(c, next)
})
app.on(privilegedMethods, '/api/page/*', async (c, next) => {
  // Single valid privileged token
  const bearer = bearerAuth({ token: privilegedToken })
  return bearer(c, next)
})

// Define handlers for GET, POST, etc.
If you want to verify the value of the token yourself, specify the verifyToken option; returning true means it is accepted.


const app = new Hono()

app.use(
  '/auth-verify-token/*',
  bearerAuth({
    verifyToken: async (token, c) => {
      return token === 'dynamic-token'
    },
  })
)
Options
required token: string | string[]
The string to validate the incoming bearer token against.

optional realm: string
The domain name of the realm, as part of the returned WWW-Authenticate challenge header. The default is "". See more: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/WWW-Authenticate#directives

optional prefix: string
The prefix (or known as schema) for the Authorization header value. The default is "Bearer".

optional headerName: string
The header name. The default value is Authorization.

optional hashFunction: Function
A function to handle hashing for safe comparison of authentication tokens.

optional verifyToken: (token: string, c: Context) => boolean | Promise<boolean>
The function to verify the token.

optional noAuthenticationHeader: object
Customizes the error response when the request does not have an authentication header.

wwwAuthenticateHeader: string | object | MessageFunction - Customizes the WWW-Authenticate header value.
message: string | object | MessageFunction - The custom message for the response body.
MessageFunction is (c: Context) => string | object | Promise<string | object>.

optional invalidAuthenticationHeader: object
Customizes the error response when the authentication header format is invalid.

wwwAuthenticateHeader: string | object | MessageFunction - Customizes the WWW-Authenticate header value.
message: string | object | MessageFunction - The custom message for the response body.
optional invalidToken: object
Customizes the error response when the token is invalid.

wwwAuthenticateHeader: string | object | MessageFunction - Customizes the WWW-Authenticate header value.
message: string | object | MessageFunction - The custom message for the response body.
Body Limit Middleware
The Body Limit Middleware can limit the file size of the request body.

This middleware first uses the value of the Content-Length header in the request, if present. If it is not set, it reads the body in the stream and executes an error handler if it is larger than the specified file size.

Import

import { Hono } from 'hono'
import { bodyLimit } from 'hono/body-limit'
Usage

const app = new Hono()

app.post(
  '/upload',
  bodyLimit({
    maxSize: 50 * 1024, // 50kb
    onError: (c) => {
      return c.text('overflow :(', 413)
    },
  }),
  async (c) => {
    const body = await c.req.parseBody()
    if (body['file'] instanceof File) {
      console.log(`Got file sized: ${body['file'].size}`)
    }
    return c.text('pass :)')
  }
)
Options
required maxSize: number
The maximum file size of the file you want to limit. The default is 100 * 1024 - 100kb.

optional onError: OnError
The error handler to be invoked if the specified file size is exceeded.

Usage with Bun for large requests
If the Body Limit Middleware is used explicitly to allow a request body larger than the default, it might be necessary to make changes to your Bun.serve configuration accordingly. At the time of writing, Bun.serve's default request body limit is 128MiB. If you set Hono's Body Limit Middleware to a value bigger than that, your requests will still fail and, additionally, the onError handler specified in the middleware will not be called. This is because Bun.serve() will set the status code to 413 and terminate the connection before passing the request to Hono.

If you want to accept requests larger than 128MiB with Hono and Bun, you need to set the limit for Bun as well:


export default {
  port: process.env['PORT'] || 3000,
  fetch: app.fetch,
  maxRequestBodySize: 1024 * 1024 * 200, // your value here
}
or, depending on your setup:


Bun.serve({
  fetch(req, server) {
    return app.fetch(req, { ip: server.requestIP(req) })
  },
  maxRequestBodySize: 1024 * 1024 * 200, // your value here
})Cache Middleware
The Cache middleware uses the Web Standards' Cache API.

The Cache middleware currently supports Cloudflare Workers projects using custom domains and Deno projects using Deno 1.26+. Also available with Deno Deploy.

Cloudflare Workers respects the Cache-Control header and return cached responses. For details, refer to Cache on Cloudflare Docs. Deno does not respect headers, so if you need to update the cache, you will need to implement your own mechanism.

See Usage below for instructions on each platform.

Import

import { Hono } from 'hono'
import { cache } from 'hono/cache'
Usage

Cloudflare Workers

Deno

app.get(
  '*',
  cache({
    cacheName: 'my-app',
    cacheControl: 'max-age=3600',
  })
)
Options
required cacheName: string | (c: Context) => string | Promise<string>
The name of the cache. Can be used to store multiple caches with different identifiers.

optional wait: boolean
A boolean indicating if Hono should wait for the Promise of the cache.put function to resolve before continuing with the request. Required to be true for the Deno environment. The default is false.

optional cacheControl: string
A string of directives for the Cache-Control header. See the MDN docs for more information. When this option is not provided, no Cache-Control header is added to requests.

optional vary: string | string[]
Sets the Vary header in the response. If the original response header already contains a Vary header, the values are merged, removing any duplicates. Setting this to * will result in an error. For more details on the Vary header and its implications for caching strategies, refer to the MDN docs.

optional keyGenerator: (c: Context) => string | Promise<string>
Generates keys for every request in the cacheName store. This can be used to cache data based on request parameters or context parameters. The default is c.req.url.

optional cacheableStatusCodes: number[]
An array of status codes that should be cached. The default is [200]. Use this option to cache responses with specific status codes.


app.get(
  '*',
  cache({
    cacheName: 'my-app',
    cacheControl: 'max-age=3600',
    cacheableStatusCodes: [200, 404, 412],
  })Compress Middleware
This middleware compresses the response body, according to Accept-Encoding request header.

INFO

Note: On Cloudflare Workers and Deno Deploy, the response body will be compressed automatically, so there is no need to use this middleware.

Import

import { Hono } from 'hono'
import { compress } from 'hono/compress'
Usage

const app = new Hono()

app.use(compress())
Options
optional encoding: 'gzip' | 'deflate'
The compression scheme to allow for response compression. Either gzip or deflate. If not defined, both are allowed and will be used based on the Accept-Encoding header. gzip is prioritized if this option is not provided and the client provides both in the Accept-Encoding header.

optional threshold: number
The minimum size in bytes to compress. Defaults to 1024 bytes.
)CORS Middleware
There are many use cases of Cloudflare Workers as Web APIs and calling them from external front-end application. For them we have to implement CORS, let's do this with middleware as well.

Import

import { Hono } from 'hono'
import { cors } from 'hono/cors'
Usage

const app = new Hono()

// CORS should be called before the route
app.use('/api/*', cors())
app.use(
  '/api2/*',
  cors({
    origin: 'http://example.com',
    allowHeaders: ['X-Custom-Header', 'Upgrade-Insecure-Requests'],
    allowMethods: ['POST', 'GET', 'OPTIONS'],
    exposeHeaders: ['Content-Length', 'X-Kuma-Revision'],
    maxAge: 600,
    credentials: true,
  })
)

app.all('/api/abc', (c) => {
  return c.json({ success: true })
})
app.all('/api2/abc', (c) => {
  return c.json({ success: true })
})
Multiple origins:


app.use(
  '/api3/*',
  cors({
    origin: ['https://example.com', 'https://example.org'],
  })
)

// Or you can use "function"
app.use(
  '/api4/*',
  cors({
    // `c` is a `Context` object
    origin: (origin, c) => {
      return origin.endsWith('.example.com')
        ? origin
        : 'http://example.com'
    },
  })
)
Dynamic allowed methods based on origin:


app.use(
  '/api5/*',
  cors({
    origin: (origin) =>
      origin === 'https://example.com' ? origin : '*',
    // `c` is a `Context` object
    allowMethods: (origin, c) =>
      origin === 'https://example.com'
        ? ['GET', 'HEAD', 'POST', 'PATCH', 'DELETE']
        : ['GET', 'HEAD'],
  })
)
Options
optional origin: string | string[] | (origin:string, c:Context) => string
The value of "Access-Control-Allow-Origin" CORS header. You can also pass the callback function like origin: (origin) => (origin.endsWith('.example.com') ? origin : 'http://example.com'). The default is *.

optional allowMethods: string[] | (origin:string, c:Context) => string[]
The value of "Access-Control-Allow-Methods" CORS header. You can also pass a callback function to dynamically determine allowed methods based on the origin. The default is ['GET', 'HEAD', 'PUT', 'POST', 'DELETE', 'PATCH'].

optional allowHeaders: string[]
The value of "Access-Control-Allow-Headers" CORS header. The default is [].

optional maxAge: number
The value of "Access-Control-Max-Age" CORS header.

optional credentials: boolean
The value of "Access-Control-Allow-Credentials" CORS header.

optional exposeHeaders: string[]
The value of "Access-Control-Expose-Headers" CORS header. The default is [].

Environment-dependent CORS configuration
If you want to adjust CORS configuration according to the execution environment, such as development or production, injecting values from environment variables is convenient as it eliminates the need for the application to be aware of its own execution environment. See the example below for clarification.


app.use('*', async (c, next) => {
  const corsMiddlewareHandler = cors({
    origin: c.env.CORS_ORIGIN,
  })
  return corsMiddlewareHandler(c, next)
})
Using with Vite
When using Hono with Vite, you should disable Vite's built-in CORS feature by setting server.cors to false in your vite.config.ts. This prevents conflicts with Hono's CORS middleware.


// vite.config.ts
import { cloudflare } from '@cloudflare/vite-plugin'
import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    cors: false, // disable Vite's built-in CORS setting
  },
  plugins: [cloudflare()],
})
Edit this page on GitHub
Last updated: 2/4/26, 3:24 AM

Pager
CSRF Protection
This middleware protects against CSRF attacks by checking both the Origin header and the Sec-Fetch-Site header. The request is allowed if either validation passes.

The middleware only validates requests that:

Use unsafe HTTP methods (not GET, HEAD, or OPTIONS)
Have content types that can be sent by HTML forms (application/x-www-form-urlencoded, multipart/form-data, or text/plain)
Old browsers that do not send Origin headers, or environments that use reverse proxies to remove these headers, may not work well. In such environments, use other CSRF token methods.

Import

import { Hono } from 'hono'
import { csrf } from 'hono/csrf'
Usage

const app = new Hono()

// Default: both origin and sec-fetch-site validation
app.use(csrf())

// Allow specific origins
app.use(csrf({ origin: 'https://myapp.example.com' }))

// Allow multiple origins
app.use(
  csrf({
    origin: [
      'https://myapp.example.com',
      'https://development.myapp.example.com',
    ],
  })
)

// Allow specific sec-fetch-site values
app.use(csrf({ secFetchSite: 'same-origin' }))
app.use(csrf({ secFetchSite: ['same-origin', 'none'] }))

// Dynamic origin validation
// It is strongly recommended that the protocol be verified to ensure a match to `$`.
// You should *never* do a forward match.
app.use(
  '*',
  csrf({
    origin: (origin) =>
      /https:\/\/(\w+\.)?myapp\.example\.com$/.test(origin),
  })
)

// Dynamic sec-fetch-site validation
app.use(
  csrf({
    secFetchSite: (secFetchSite, c) => {
      // Always allow same-origin
      if (secFetchSite === 'same-origin') return true
      // Allow cross-site for webhook endpoints
      if (
        secFetchSite === 'cross-site' &&
        c.req.path.startsWith('/webhook/')
      ) {
        return true
      }
      return false
    },
  })
)
Options
optional origin: string | string[] | Function
Specify allowed origins for CSRF protection.

string: Single allowed origin (e.g., 'https://example.com')
string[]: Array of allowed origins
Function: Custom handler (origin: string, context: Context) => boolean for flexible origin validation and bypass logic
Default: Only same origin as the request URL

The function handler receives the request's Origin header value and the request context, allowing for dynamic validation based on request properties like path, headers, or other context data.

optional secFetchSite: string | string[] | Function
Specify allowed Sec-Fetch-Site header values for CSRF protection using Fetch Metadata.

string: Single allowed value (e.g., 'same-origin')
string[]: Array of allowed values (e.g., ['same-origin', 'none'])
Function: Custom handler (secFetchSite: string, context: Context) => boolean for flexible validation
Default: Only allows 'same-origin'

Standard Sec-Fetch-Site values:

same-origin: Request from same origin
same-site: Request from same site (different subdomain)
cross-site: Request from different site
none: Request not from a web page (e.g., browser address bar, bookmark)
The function handler receives the request's Sec-Fetch-Site header value and the request context, enabling dynamic validation based on request JWT Auth Middleware
The JWT Auth Middleware provides authentication by verifying the token with JWT. The middleware will check for an Authorization header if the cookie option is not set. You can customize the header name using the headerName option.

INFO

The Authorization header sent from the client must have a specified scheme.

Example: Bearer my.token.value or Basic my.token.value

Import

import { Hono } from 'hono'
import { jwt } from 'hono/jwt'
import type { JwtVariables } from 'hono/jwt'
Usage

// Specify the variable types to infer the `c.get('jwtPayload')`:
type Variables = JwtVariables

const app = new Hono<{ Variables: Variables }>()

app.use(
  '/auth/*',
  jwt({
    secret: 'it-is-very-secret',
    alg: 'HS256',
  })
)

app.get('/auth/page', (c) => {
  return c.text('You are authorized')
})
Get payload:


const app = new Hono()

app.use(
  '/auth/*',
  jwt({
    secret: 'it-is-very-secret',
    alg: 'HS256',
    issuer: 'my-trusted-issuer',
  })
)

app.get('/auth/page', (c) => {
  const payload = c.get('jwtPayload')
  return c.json(payload) // eg: { "sub": "1234567890", "name": "John Doe", "iat": 1516239022, "iss": "my-trusted-issuer" }
})
TIP

jwt() is just a middleware function. If you want to use an environment variable (eg: c.env.JWT_SECRET), you can use it as follows:


app.use('/auth/*', (c, next) => {
  const jwtMiddleware = jwt({
    secret: c.env.JWT_SECRET,
    alg: 'HS256',
  })
  return jwtMiddleware(c, next)
})
Options
required secret: string
A value of your secret key.

required alg: string
An algorithm type that is used for verifying.

Available types are HS256 | HS384 | HS512 | RS256 | RS384 | RS512 | PS256 | PS384 | PS512 | ES256 | ES384 | ES512 | EdDSA.

optional cookie: string
If this value is set, then the value is retrieved from the cookie header using that value as a key, which is then validated as a token.

optional headerName: string
The name of the header to look for the JWT token. The default is Authorization.


app.use(
  '/auth/*',
  jwt({
    secret: 'it-is-very-secret',
    alg: 'HS256',
    headerName: 'x-custom-auth-header',
  })
)
optional verifyOptions: VerifyOptions
Options controlling verification of the token.

optional verifyOptions.iss: string | RexExp
The expected issuer used for token verification. The iss claim will not be checked if this isn't set.

optional verifyOptions.nbf: boolean
The nbf (not before) claim will be verified if present and this is set to true. The default is true.

optional verifyOptions.iat: boolean
The iat (not before) claim will be verified if present and this is set to true. The default is true.

optional verifyOptions.exp: boolean
The exp (not before) claim will be verified if present and this is set to true. The default is true.
Logger Middleware
It's a simple logger.

Import

import { Hono } from 'hono'
import { logger } from 'hono/logger'
Usage

const app = new Hono()

app.use(logger())
app.get('/', (c) => c.text('Hello Hono!'))
Logging Details
The Logger Middleware logs the following details for each request:

Incoming Request: Logs the HTTP method, request path, and incoming request.
Outgoing Response: Logs the HTTP method, request path, response status code, and request/response times.
Status Code Coloring: Response status codes are color-coded for better visibility and quick identification of status categories. Different status code categories are represented by different colors.
Elapsed Time: The time taken for the request/response cycle is logged in a human-readable format, either in milliseconds (ms) or seconds (s).
By using the Logger Middleware, you can easily monitor the flow of requests and responses in your Hono application and quickly identify any issues or performance bottlenecks.

You can also extend the middleware further by providing your own PrintFunc function for tailored logging behavior.

PrintFunc
The Logger Middleware accepts an optional PrintFunc function as a parameter. This function allows you to customize the logger and add additional logs.

Options
optional fn: PrintFunc(str: string, ...rest: string[])
str: Passed by the logger.
...rest: Additional string props to be printed to console.
Example
Setting up a custom PrintFunc function to the Logger Middleware:


export const customLogger = (message: string, ...rest: string[]) => {
  console.log(message, ...rest)
}

app.use(logger(customLogger))
Setting up the custom logger in a route:


app.post('/blog', (c) => {
  // Routing logic

  customLogger('Blog saved:', `Path: ${blog.url},`, `ID: ${blog.id}`)
  // Output
  // <-- POST /blog
  // Blog saved: Path: /blog/example, ID: 1
  // --> POST /blog 201 93ms

  // Return Context
})Pretty JSON Middleware
Pretty JSON middleware enables "JSON pretty print" for JSON response body. Adding ?pretty to url query param, the JSON strings are prettified.


// GET /
{"project":{"name":"Hono","repository":"https://github.com/honojs/hono"}}
will be:


// GET /?pretty
{
  "project": {
    "name": "Hono",
    "repository": "https://github.com/honojs/hono"
  }
}
Import

import { Hono } from 'hono'
import { prettyJSON } from 'hono/pretty-json'
Usage

const app = new Hono()

app.use(prettyJSON()) // With options: prettyJSON({ space: 4 })
app.get('/', (c) => {
  return c.json({ message: 'Hono!' })
})
Options
optional space: number
Number of spaces for indentation. The default is 2.

optional query: string
The name of the query string for applying. The default is pretty.

optional force: boolean
When set to true, JSON responses are always prettified regardless of the query parameter. The default is false.
Request ID Middleware
Request ID Middleware generates a unique ID for each request, which you can use in your handlers.

INFO

Node.js: This middleware uses crypto.randomUUID() to generate IDs. The global crypto was introduced in Node.js version 20 or later. Therefore, errors may occur in versions earlier than that. In that case, please specify generator. However, if you are using the Node.js adapter, it automatically sets crypto globally, so this is not necessary.

Import

import { Hono } from 'hono'
import { requestId } from 'hono/request-id'
Usage
You can access the Request ID through the requestId variable in the handlers and middleware to which the Request ID Middleware is applied.


const app = new Hono()

app.use('*', requestId())

app.get('/', (c) => {
  return c.text(`Your request id is ${c.get('requestId')}`)
})
If you want to explicitly specify the type, import RequestIdVariables and pass it in the generics of new Hono().


import type { RequestIdVariables } from 'hono/request-id'

const app = new Hono<{
  Variables: RequestIdVariables
}>()
Set Request ID
You set a custom request ID in the header (default: X-Request-Id), the middleware will use that value instead of generating a new one:


const app = new Hono()

app.use('*', requestId())

app.get('/', (c) => {
  return c.text(`${c.get('requestId')}`)
})

const res = await app.request('/', {
  headers: {
    'X-Request-Id': 'your-custom-id',
  },
})
console.log(await res.text()) // your-custom-id
If you want to disable this feature, set headerName option to an empty string.

Options
optional limitLength: number
The maximum length of the request ID. The default is 255.

optional headerName: string
The header name used for the request ID. The default is X-Request-Id.

optional generator: (c: Context) => string
The request ID generation function. By default, it uses crypto.randomUUID().

Platform specific Request IDs
Some platform (such as AWS Lambda) already generate their own Request IDs per request. Without any additional configuration, this middleware is unaware of these specific Request IDs and generates a new Request ID. This can lead to confusion when looking at your application logs.

To unify these IDs, use the generator function to capture the platform specific Request ID and to use it in this middleware.

Platform specific links
AWS Lambda
AWS documentation: Context object
Hono: Access AWS Lambda Object
Cloudflare
Cloudflare Ray ID
Deno
Request ID on the Deno Blog
Fastly
Fastly documentation: req.xid
Secure Headers Middleware
Secure Headers Middleware simplifies the setup of security headers. Inspired in part by the capabilities of Helmet, it allows you to control the activation and deactivation of specific security headers.

Import

import { Hono } from 'hono'
import { secureHeaders } from 'hono/secure-headers'
Usage
You can use the optimal settings by default.


const app = new Hono()
app.use(secureHeaders())
You can suppress unnecessary headers by setting them to false.


const app = new Hono()
app.use(
  '*',
  secureHeaders({
    xFrameOptions: false,
    xXssProtection: false,
  })
)
You can override default header values using a string.


const app = new Hono()
app.use(
  '*',
  secureHeaders({
    strictTransportSecurity:
      'max-age=63072000; includeSubDomains; preload',
    xFrameOptions: 'DENY',
    xXssProtection: '1',
  })
)
Supported Options
Each option corresponds to the following Header Key-Value pairs.

Option	Header	Value	Default
-	X-Powered-By	(Delete Header)	True
contentSecurityPolicy	Content-Security-Policy	Usage: Setting Content-Security-Policy	No Setting
contentSecurityPolicyReportOnly	Content-Security-Policy-Report-Only	Usage: Setting Content-Security-Policy	No Setting
trustedTypes	Trusted Types	Usage: Setting Content-Security-Policy	No Setting
requireTrustedTypesFor	Require Trusted Types For	Usage: Setting Content-Security-Policy	No Setting
crossOriginEmbedderPolicy	Cross-Origin-Embedder-Policy	require-corp	False
crossOriginResourcePolicy	Cross-Origin-Resource-Policy	same-origin	True
crossOriginOpenerPolicy	Cross-Origin-Opener-Policy	same-origin	True
originAgentCluster	Origin-Agent-Cluster	?1	True
referrerPolicy	Referrer-Policy	no-referrer	True
reportingEndpoints	Reporting-Endpoints	Usage: Setting Content-Security-Policy	No Setting
reportTo	Report-To	Usage: Setting Content-Security-Policy	No Setting
strictTransportSecurity	Strict-Transport-Security	max-age=15552000; includeSubDomains	True
xContentTypeOptions	X-Content-Type-Options	nosniff	True
xDnsPrefetchControl	X-DNS-Prefetch-Control	off	True
xDownloadOptions	X-Download-Options	noopen	True
xFrameOptions	X-Frame-Options	SAMEORIGIN	True
xPermittedCrossDomainPolicies	X-Permitted-Cross-Domain-Policies	none	True
xXssProtection	X-XSS-Protection	0	True
permissionPolicy	Permissions-Policy	Usage: Setting Permission-Policy	No Setting
Middleware Conflict
Please be cautious about the order of specification when dealing with middleware that manipulates the same header.

In this case, Secure-headers operates and the x-powered-by is removed:


const app = new Hono()
app.use(secureHeaders())
app.use(poweredBy())
In this case, Powered-By operates and the x-powered-by is added:


const app = new Hono()
app.use(poweredBy())
app.use(secureHeaders())
Setting Content-Security-Policy

const app = new Hono()
app.use(
  '/test',
  secureHeaders({
    reportingEndpoints: [
      {
        name: 'endpoint-1',
        url: 'https://example.com/reports',
      },
    ],
    // -- or alternatively
    // reportTo: [
    //   {
    //     group: 'endpoint-1',
    //     max_age: 10886400,
    //     endpoints: [{ url: 'https://example.com/reports' }],
    //   },
    // ],
    contentSecurityPolicy: {
      defaultSrc: ["'self'"],
      baseUri: ["'self'"],
      childSrc: ["'self'"],
      connectSrc: ["'self'"],
      fontSrc: ["'self'", 'https:', 'data:'],
      formAction: ["'self'"],
      frameAncestors: ["'self'"],
      frameSrc: ["'self'"],
      imgSrc: ["'self'", 'data:'],
      manifestSrc: ["'self'"],
      mediaSrc: ["'self'"],
      objectSrc: ["'none'"],
      reportTo: 'endpoint-1',
      reportUri: '/csp-report',
      sandbox: ['allow-same-origin', 'allow-scripts'],
      scriptSrc: ["'self'"],
      scriptSrcAttr: ["'none'"],
      scriptSrcElem: ["'self'"],
      styleSrc: ["'self'", 'https:', "'unsafe-inline'"],
      styleSrcAttr: ['none'],
      styleSrcElem: ["'self'", 'https:', "'unsafe-inline'"],
      upgradeInsecureRequests: [],
      workerSrc: ["'self'"],
    },
  })
)
nonce attribute
You can add a nonce attribute to a script or style element by adding the NONCE imported from hono/secure-headers to a scriptSrc or styleSrc:


import { secureHeaders, NONCE } from 'hono/secure-headers'
import type { SecureHeadersVariables } from 'hono/secure-headers'

// Specify the variable types to infer the `c.get('secureHeadersNonce')`:
type Variables = SecureHeadersVariables

const app = new Hono<{ Variables: Variables }>()

// Set the pre-defined nonce value to `scriptSrc`:
app.get(
  '*',
  secureHeaders({
    contentSecurityPolicy: {
      scriptSrc: [NONCE, 'https://allowed1.example.com'],
    },
  })
)

// Get the value from `c.get('secureHeadersNonce')`:
app.get('/', (c) => {
  return c.html(
    <html>
      <body>
        {/** contents */}
        <script
          src='/js/client.js'
          nonce={c.get('secureHeadersNonce')}
        />
      </body>
    </html>
  )
})
If you want to generate the nonce value yourself, you can also specify a function as the following:


const app = new Hono<{
  Variables: { myNonce: string }
}>()

const myNonceGenerator: ContentSecurityPolicyOptionHandler = (c) => {
  // This function is called on every request.
  const nonce = Math.random().toString(36).slice(2)
  c.set('myNonce', nonce)
  return `'nonce-${nonce}'`
}

app.get(
  '*',
  secureHeaders({
    contentSecurityPolicy: {
      scriptSrc: [myNonceGenerator, 'https://allowed1.example.com'],
    },
  })
)

app.get('/', (c) => {
  return c.html(
    <html>
      <body>
        {/** contents */}
        <script src='/js/client.js' nonce={c.get('myNonce')} />
      </body>
    </html>
  )
})
Setting Permission-Policy
The Permission-Policy header allows you to control which features and APIs can be used in the browser. Here's an example of how to set it:


const app = new Hono()
app.use(
  '*',
  secureHeaders({
    permissionsPolicy: {
      fullscreen: ['self'], // fullscreen=(self)
      bluetooth: ['none'], // bluetooth=(none)
      payment: ['self', 'https://example.com'], // payment=(self "https://example.com")
      syncXhr: [], // sync-xhr=()
      camera: false, // camera=none
      microphone: true, // microphone=*
      geolocation: ['*'], // geolocation=*
      usb: ['self', 'https://a.example.com', 'https://b.example.com'], // usb=(self "https://a.example.com" "https://b.example.com")
      accelerometer: ['https://*.example.com'], // accelerometer=("https://*.example.com")
      gyroscope: ['src'], // gyroscope=(src)
      magnetometer: [
        'https://a.example.com',
        'https://b.example.com',
      ], // magnetometer=("https://a.example.com" "https://b.example.com")
    },
  })
)
Edit this page Timeout Middleware
The Timeout Middleware enables you to easily manage request timeouts in your application. It allows you to set a maximum duration for requests and optionally define custom error responses if the specified timeout is exceeded.

Import

import { Hono } from 'hono'
import { timeout } from 'hono/timeout'
Usage
Here's how to use the Timeout Middleware with both default and custom settings:

Default Settings:


const app = new Hono()

// Applying a 5-second timeout
app.use('/api', timeout(5000))

// Handling a route
app.get('/api/data', async (c) => {
  // Your route handler logic
  return c.json({ data: 'Your data here' })
})
Custom settings:


import { HTTPException } from 'hono/http-exception'

// Custom exception factory function
const customTimeoutException = (context) =>
  new HTTPException(408, {
    message: `Request timeout after waiting ${context.req.headers.get(
      'Duration'
    )} seconds. Please try again later.`,
  })

// for Static Exception Message
// const customTimeoutException = new HTTPException(408, {
//   message: 'Operation timed out. Please try again later.'
// });

// Applying a 1-minute timeout with a custom exception
app.use('/api/long-process', timeout(60000, customTimeoutException))

app.get('/api/long-process', async (c) => {
  // Simulate a long process
  await new Promise((resolve) => setTimeout(resolve, 61000))
  return c.json({ data: 'This usually takes longer' })
})
Notes
The duration for the timeout can be specified in milliseconds. The middleware will automatically reject the promise and potentially throw an error if the specified duration is exceeded.

The timeout middleware cannot be used with stream Thus, use stream.close and setTimeout together.


app.get('/sse', async (c) => {
  let id = 0
  let running = true
  let timer: number | undefined

  return streamSSE(c, async (stream) => {
    timer = setTimeout(() => {
      console.log('Stream timeout reached, closing stream')
      stream.close()
    }, 3000) as unknown as number

    stream.onAbort(async () => {
      console.log('Client closed connection')
      running = false
      clearTimeout(timer)
    })

    while (running) {
      const message = `It is ${new Date().toISOString()}`
      await stream.writeSSE({
        data: message,
        event: 'time-update',
        id: String(id++),
      })
      await stream.sleep(1000)
    }
  })
})
Middleware Conflicts
Be cautious about the order of middleware, especially when using error-handling or other timing-related middleware, as it might affect the behavior of this timeout middleware.
Third-party Middleware
Third-party middleware refers to middleware not bundled within the Hono package. Most of this middleware leverages external libraries.

Authentication
Auth.js(Next Auth)
Casbin
Clerk Auth
Cloudflare Access
OAuth Providers
OIDC Auth
Firebase Auth
Verify RSA JWT (JWKS)
Stytch Auth
Validators
Ajv Validator
ArkType Validator
Class Validator
Conform Validator
Effect Schema Validator
Standard Schema Validator
TypeBox Validator
Typia Validator
unknownutil Validator
Valibot Validator
Zod Validator
OpenAPI
Zod OpenAPI
Scalar
Swagger UI
Swagger Editor
Hono OpenAPI
hono-zod-openapi
Development
ESLint Config
SSG Plugin Essential
Monitoring / Tracing
Apitally (API monitoring & analytics)
Highlight.io
LogTape (Logging)
OpenTelemetry
Prometheus Metrics
Sentry
Server / Adapter
GraphQL Server
Node WebSocket Helper
tRPC Server
Transpiler
Bun Transpiler
esbuild Transpiler
UI / Renderer
Qwik City
React Compatibility
React Renderer
Utilities
Bun Compress
Cap Checkpoint
Event Emitter
Geo
Hono Rate Limiter
Hono Simple DI
jsonv-ts (Validator, OpenAPI, MCP)
MCP
RONIN (Database)
Session
tsyringe
User Agent based Blocker
Trailing Slash Middleware
This middleware handles Trailing Slash in the URL on a GET request.

appendTrailingSlash redirects the URL to which it added the Trailing Slash if the content was not found. Also, trimTrailingSlash will remove the Trailing Slash.

Import

import { Hono } from 'hono'
import {
  appendTrailingSlash,
  trimTrailingSlash,
} from 'hono/trailing-slash'
Usage
Example of redirecting a GET request of /about/me to /about/me/.


import { Hono } from 'hono'
import { appendTrailingSlash } from 'hono/trailing-slash'

const app = new Hono({ strict: true })

app.use(appendTrailingSlash())
app.get('/about/me/', (c) => c.text('With Trailing Slash'))
Example of redirecting a GET request of /about/me/ to /about/me.


import { Hono } from 'hono'
import { trimTrailingSlash } from 'hono/trailing-slash'

const app = new Hono({ strict: true })

app.use(trimTrailingSlash())
app.get('/about/me', (c) => c.text('Without Trailing Slash'))
Note
It will be enabled when the request method is GET and the resp



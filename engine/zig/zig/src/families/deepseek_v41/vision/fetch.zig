//! Image URLs as prod fetches them (GLM's ``image_fetch.py``, which DeepSeek's ``vision_prep.load_bytes`` calls):
//! ``https`` on port 443 (TF_DSV41_VISION_FETCH_HTTP=1: also ``http`` and any port) with a host, no credentials,
//! fragment, backslash, whitespace or control characters, at most 4,096 characters; the host resolved and **every**
//! address public (``public_ip``: Python's ``ipaddress.is_global`` rule, not multicast / reserved / 6to4 / Teredo /
//! IPv4-mapped / NAT64 / IPv4-compatible, not special-purpose networks or platform metadata endpoints; TF_DSV41_VISION_FETCH_PRIVATE=1: any);
//! the socket connected to a checked address and TLS verified for the host name (no second lookup); ``identity``
//! encoding, a declared JPEG / PNG / WebP / GIF, at most TF_DSV41_VISION_MAX_BYTES; each redirect (at most 3) checked
//! from the start; one deadline over everything (TF_DSV41_VISION_FETCH_TIMEOUT a download within the request's
//! TF_DSV41_VISION_FETCH_TOTAL_S), a watchdog shutting the socket when it passes. Messages never hold the URL.
//! Differences: the connection is pinned to the first IPv4 address (an IPv6-only host is refused: std's client
//! takes an IPv4 literal as the pinned host), and an internationalized host name is refused (no IDNA).
const std = @import("std");
const prep = @import("prep.zig");
const parts = @import("parts.zig");
const host_mod = @import("host.zig");
const Allocator = std.mem.Allocator;
const net = std.Io.net;

pub const Policy = struct {
    http: bool = false,
    private: bool = false,
    max_redirects: u32 = 3,
    max_url_chars: usize = 4096,
};

pub const user_agent = "tensorfold-glm53-vision/2";
const media_types = [_][]const u8{ "image/jpeg", "image/png", "image/webp", "image/gif" };
const blocked_hosts = [_][]const u8{ "localhost", "metadata.google.internal", "instance-data", "metadata" };

/// One hop's target: scheme, lowercase host, port, the quoted request target.
pub const Target = struct { https: bool, host: []const u8, port: u16, target: []const u8 };

fn refuse(problem: *parts.Problem, msg: []const u8) error{Refused} {
    problem.* = .{ .message = msg };
    return error.Refused;
}

fn in4(a: [4]u8, net4: [4]u8, bits: u5) bool {
    const x = std.mem.readInt(u32, &a, .big);
    const y = std.mem.readInt(u32, &net4, .big);
    const m: u32 = if (bits == 0) 0 else ~@as(u32, 0) << @intCast(32 - @as(u6, bits));
    return x & m == y & m;
}

fn in6(a: [16]u8, net6: u128, bits: u8) bool {
    const x = std.mem.readInt(u128, &a, .big);
    const m: u128 = if (bits == 0) 0 else ~@as(u128, 0) << @intCast(128 - @as(u8, bits));
    return x & m == net6 & m;
}

/// An IPv6 network's first 16-bit groups (the rest zero), as one big-endian u128.
fn v6(comptime groups: []const u16) u128 {
    var x: u128 = 0;
    for (groups, 0..) |g, i| x |= @as(u128, g) << @intCast(112 - 16 * i);
    return x;
}

/// ``public_ip``: Python 3.12's ``ipaddress`` rules (``is_global``, ``is_multicast``, ``is_reserved``) plus
/// image_fetch's own blocks.
pub fn publicIp(addr: net.IpAddress) bool {
    switch (addr) {
        .ip4 => |a4| {
            const a = a4.bytes;
            // is_private (3.12's table); is_global: not private and not the shared-address range
            const private = [_]struct { [4]u8, u5 }{
                .{ .{ 0, 0, 0, 0 }, 8 },      .{ .{ 10, 0, 0, 0 }, 8 },     .{ .{ 127, 0, 0, 0 }, 8 },     .{ .{ 169, 254, 0, 0 }, 16 },
                .{ .{ 172, 16, 0, 0 }, 12 },  .{ .{ 192, 0, 0, 0 }, 29 },   .{ .{ 192, 0, 0, 170 }, 31 },  .{ .{ 192, 0, 2, 0 }, 24 },
                .{ .{ 192, 168, 0, 0 }, 16 }, .{ .{ 198, 18, 0, 0 }, 15 },  .{ .{ 198, 51, 100, 0 }, 24 }, .{ .{ 203, 0, 113, 0 }, 24 },
                .{ .{ 240, 0, 0, 0 }, 4 },    .{ .{ 255, 255, 255, 255 }, 0 },
            };
            for (private) |p| if (p[1] == 0) {
                if (std.mem.eql(u8, &a, &p[0])) return false;
            } else if (in4(a, p[0], p[1])) return false;
            if (in4(a, .{ 100, 64, 0, 0 }, 10)) return false;
            if (in4(a, .{ 224, 0, 0, 0 }, 4)) return false; // multicast
            if (in4(a, .{ 192, 0, 0, 0 }, 24)) return false; // _SPECIAL_V4
            if (std.mem.eql(u8, &a, &.{ 168, 63, 129, 16 })) return false;
            return true;
        },
        .ip6 => |a6| {
            const a = a6.bytes;
            const zero = std.mem.allEqual(u8, &a, 0);
            const loop = std.mem.allEqual(u8, a[0..15], 0) and a[15] == 1;
            if (zero or loop) return false;
            const private = [_]struct { u128, u8 }{
                .{ (0xffff << 32), 96 }, .{ v6(&.{ 0x64, 0xff9b, 1 }), 48 }, .{ v6(&.{0x100}), 64 }, .{ v6(&.{ 0x2001, 0xdb8 }), 32 },
                .{ v6(&.{ 0x2001, 0x10 }), 28 },  .{ v6(&.{0xfc00}), 7 },        .{ v6(&.{0xfe80}), 10 },
            };
            for (private) |p| if (in6(a, p[0], p[1])) return false;
            // 2001::/23 is private except these (3.12)
            if (in6(a, v6(&.{0x2001}), 23)) {
                const ok = [_]struct { u128, u8 }{ .{ (v6(&.{ 0x2001, 1 }) | 1), 128 }, .{ (v6(&.{ 0x2001, 1 }) | 2), 128 }, .{ v6(&.{ 0x2001, 3 }), 32 }, .{ v6(&.{ 0x2001, 4, 0x112 }), 48 }, .{ v6(&.{ 0x2001, 0x20 }), 28 }, .{ v6(&.{ 0x2001, 0x30 }), 28 } };
                const exempt = for (ok) |e| {
                    if (in6(a, e[0], e[1])) break true;
                } else false;
                if (!exempt) return false;
            }
            if (a[0] == 0xff) return false; // multicast
            // is_reserved: ::/8 (but not ::/128 / ::1 handled above) and the unassigned blocks
            const reserved = [_]struct { u128, u8 }{
                .{ 0, 8 },     .{ v6(&.{0x100}), 8 },  .{ v6(&.{0x200}), 7 },  .{ v6(&.{0x400}), 6 },  .{ v6(&.{0x800}), 5 },
                .{ v6(&.{0x1000}), 4 }, .{ v6(&.{0x4000}), 3 }, .{ v6(&.{0x6000}), 3 }, .{ v6(&.{0x8000}), 3 }, .{ v6(&.{0xa000}), 3 },
                .{ v6(&.{0xc000}), 3 }, .{ v6(&.{0xe000}), 4 }, .{ v6(&.{0xf000}), 5 }, .{ v6(&.{0xf800}), 6 }, .{ v6(&.{0xfe00}), 9 },
            };
            for (reserved) |p| if (in6(a, p[0], p[1])) return false;
            if (in6(a, v6(&.{0x2002}), 16)) return false; // sixtofour
            if (in6(a, v6(&.{0x2001}), 32)) return false; // teredo
            if (in6(a, v6(&.{ 0x64, 0xff9b }), 96) or in6(a, 0, 96)) return false; // NAT64, IPv4-compatible
            return true;
        },
    }
}

/// ``check_url``: the hop's target, or the refusal Python gives.
pub fn checkUrl(a: Allocator, url: []const u8, policy: Policy, problem: *parts.Problem) error{ Refused, OutOfMemory }!Target {
    if (url.len > policy.max_url_chars) return refuse(problem, "image URL is too long or contains whitespace/control characters");
    for (url) |c| if (c <= 32 or c == 127) return refuse(problem, "image URL is too long or contains whitespace/control characters");
    const rule = if (policy.http) "image URL must be HTTP(S), without credentials or a fragment" else "image URL must be HTTPS on port 443, without credentials or a fragment";
    const colon = std.mem.indexOfScalar(u8, url, ':') orelse return refuse(problem, rule);
    const scheme = url[0..colon];
    const https = std.ascii.eqlIgnoreCase(scheme, "https");
    if (!https and !(policy.http and std.ascii.eqlIgnoreCase(scheme, "http"))) return refuse(problem, rule);
    if (std.mem.indexOfScalar(u8, url, '\\') != null) return refuse(problem, rule);
    var rest = url[colon + 1 ..];
    if (!std.mem.startsWith(u8, rest, "//")) return refuse(problem, rule);
    rest = rest[2..];
    if (std.mem.indexOfScalar(u8, rest, '#')) |i| { // urlsplit's fragment: refused when not empty
        if (i + 1 < rest.len) return refuse(problem, rule);
        rest = rest[0..i];
    }
    const end = std.mem.indexOfAny(u8, rest, "/?") orelse rest.len;
    const netloc = rest[0..end];
    const tail = rest[end..];
    if (std.mem.indexOfScalar(u8, netloc, '@') != null) return refuse(problem, rule);
    var hostpart = netloc;
    var portpart: []const u8 = "";
    if (std.mem.startsWith(u8, netloc, "[")) {
        const close = std.mem.indexOfScalar(u8, netloc, ']') orelse return refuse(problem, rule);
        hostpart = netloc[1..close];
        const after = netloc[close + 1 ..];
        if (after.len > 0) {
            if (after[0] != ':') return refuse(problem, rule);
            portpart = after[1..];
        }
    } else if (std.mem.lastIndexOfScalar(u8, netloc, ':')) |c| {
        hostpart = netloc[0..c];
        portpart = netloc[c + 1 ..];
    }
    if (hostpart.len == 0 or std.mem.indexOfScalar(u8, hostpart, '%') != null) return refuse(problem, rule);
    for (hostpart) |c| if (c >= 0x80) return refuse(problem, "image URL host names must be ASCII (internationalized names are not fetched)");
    const host = try std.ascii.allocLowerString(a, hostpart);
    var port: u16 = if (https) 443 else 80;
    if (portpart.len > 0) port = std.fmt.parseInt(u16, portpart, 10) catch return refuse(problem, rule);
    if (!policy.http and port != 443) return refuse(problem, rule);
    if (!policy.private) {
        const h = std.mem.trimEnd(u8, host, ".");
        for (blocked_hosts) |b| if (std.mem.eql(u8, h, b)) return refuse(problem, "image URLs must use public internet hosts");
    }
    // the target: quote(path or "/", safe=...) + "?" + quote(query, safe=... + "?")
    var t: std.ArrayList(u8) = .empty;
    const q = std.mem.indexOfScalar(u8, tail, '?');
    const path = if (q) |i| tail[0..i] else tail;
    try quote(a, &t, if (path.len == 0) "/" else path, "/%:@!$&'()*+,;=-._~");
    if (q) |i| if (tail.len > i + 1) {
        try t.append(a, '?');
        try quote(a, &t, tail[i + 1 ..], "/%?:@!$&'()*+,;=-._~");
    };
    return .{ .https = https, .host = host, .port = port, .target = t.items };
}

/// ``urllib.parse.quote(s, safe)`` of a UTF-8 string: unreserved and safe bytes kept, the rest %XX (upper case).
fn quote(a: Allocator, out: *std.ArrayList(u8), s: []const u8, safe: []const u8) Allocator.Error!void {
    for (s) |c| {
        if (std.ascii.isAlphanumeric(c) or std.mem.indexOfScalar(u8, "_.-~", c) != null or std.mem.indexOfScalar(u8, safe, c) != null) {
            try out.append(a, c);
        } else try out.print(a, "%{X:0>2}", .{c});
    }
}

/// A fetcher for ``host.Host``: ``Host.fetcher = fetcher.it()`` (rank 0's server).
pub const Fetcher = struct {
    gpa: Allocator,
    io: std.Io,
    policy: Policy,
    client: std.http.Client,
    dns_slots: std.atomic.Value(u32) = .init(0),

    pub fn init(gpa: Allocator, io: std.Io, s: *const prep.Settings) Fetcher {
        return .{ .gpa = gpa, .io = io, .policy = .{ .http = s.fetch_http, .private = s.fetch_private }, .client = .{ .allocator = gpa, .io = io } };
    }

    pub fn deinit(f: *Fetcher) void {
        f.client.deinit();
    }

    pub fn it(f: *Fetcher) host_mod.Fetcher {
        return .{ .ctx = f, .fetch = fetchFn };
    }

    fn fetchFn(ctx: *anyopaque, a: Allocator, url: []const u8, s: *const prep.Settings, deadline_ns: i96, problem: *parts.Problem) error{ Refused, OutOfMemory }![]u8 {
        const f: *Fetcher = @ptrCast(@alignCast(ctx));
        const now = std.Io.Clock.awake.now(f.io).nanoseconds;
        const until = @min(deadline_ns, now + @as(i96, @intFromFloat(s.fetch_timeout_s * 1e9)));
        return f.fetch(a, url, s.max_bytes, until, problem);
    }

    fn remaining(f: *Fetcher, deadline: i96, problem: *parts.Problem) error{Refused}!i96 {
        const left = deadline - std.Io.Clock.awake.now(f.io).nanoseconds;
        if (left <= 0) return refuse(problem, "image download timed out");
        return left;
    }

    /// ``fetch_image``: every hop checked, resolved, its addresses checked, fetched from a checked address.
    pub fn fetch(f: *Fetcher, a: Allocator, url0: []const u8, max_bytes: usize, deadline: i96, problem: *parts.Problem) error{ Refused, OutOfMemory }![]u8 {
        var url = url0;
        var hop: u32 = 0;
        while (hop <= f.policy.max_redirects) : (hop += 1) {
            const t = try checkUrl(a, url, f.policy, problem);
            const addr = try f.resolve(t, deadline, problem);
            const got = try f.request(a, t, addr, max_bytes, deadline, problem);
            switch (got) {
                .body => |b| return b,
                .location => |loc| url = try resolveRef(a, t, url, loc),
            }
        }
        return refuse(problem, "image URL redirected too many times");
    }

    /// The host's addresses (every one public unless the policy says private); the first IPv4 one to connect to.
    fn resolve(f: *Fetcher, t: Target, deadline: i96, problem: *parts.Problem) error{ Refused, OutOfMemory }!net.IpAddress {
        if (net.IpAddress.parse(t.host, t.port)) |lit| {
            if (!f.policy.private and !publicIp(lit)) return refuse(problem, "image URLs must resolve only to public internet addresses");
            if (lit != .ip4) return refuse(problem, "image hosts reachable only over IPv6 are not fetched by this engine");
            return lit;
        } else |_| {}
        // ``_DNS_SLOTS``: at most 4 lookups at once, waiting for one until the deadline
        while (true) {
            const cur = f.dns_slots.load(.acquire);
            if (cur < 4 and f.dns_slots.cmpxchgWeak(cur, cur + 1, .acq_rel, .acquire) == null) break;
            _ = try f.remaining(deadline, problem);
            std.Io.sleep(f.io, .fromMilliseconds(5), .awake) catch {};
        }
        defer _ = f.dns_slots.fetchSub(1, .acq_rel);
        const name = net.HostName.init(t.host) catch return refuse(problem, "image host could not be resolved");
        var buf: [32]net.HostName.LookupResult = undefined;
        var queue: std.Io.Queue(net.HostName.LookupResult) = .init(&buf);
        name.lookup(f.io, &queue, .{ .port = t.port }) catch return refuse(problem, "image host could not be resolved");
        _ = try f.remaining(deadline, problem);
        var first4: ?net.IpAddress = null;
        var any = false;
        while (queue.getOneUncancelable(f.io) catch null) |r| switch (r) {
            .address => |ad| {
                any = true;
                if (!f.policy.private and !publicIp(ad)) return refuse(problem, "image URLs must resolve only to public internet addresses");
                if (first4 == null and ad == .ip4) first4 = ad;
            },
            .canonical_name => break,
        };
        if (!any) return refuse(problem, "image host could not be resolved");
        return first4 orelse refuse(problem, "image hosts reachable only over IPv6 are not fetched by this engine");
    }

    const Got = union(enum) { body: []u8, location: []const u8 };

    /// One GET on the checked address (TLS for the host name), the watchdog closing it at the deadline.
    fn request(f: *Fetcher, a: Allocator, t: Target, addr: net.IpAddress, max_bytes: usize, deadline: i96, problem: *parts.Problem) error{ Refused, OutOfMemory }!Got {
        const left = try f.remaining(deadline, problem);
        var lit_buf: [64]u8 = undefined;
        const lit = std.fmt.bufPrint(&lit_buf, "{d}.{d}.{d}.{d}", .{ addr.ip4.bytes[0], addr.ip4.bytes[1], addr.ip4.bytes[2], addr.ip4.bytes[3] }) catch unreachable;
        const pinned = net.HostName.init(lit) catch unreachable;
        const named = net.HostName.init(t.host) catch return refuse(problem, "image host could not be resolved");
        const conn = f.client.connectTcpOptions(.{
            .host = pinned,
            .port = t.port,
            .protocol = if (t.https) .tls else .plain,
            .proxied_host = named,
            .proxied_port = t.port,
            .timeout = .{ .duration = .{ .raw = .fromNanoseconds(left), .clock = .awake } },
        }) catch return refuse(problem, "could not fetch the image URL: the connection failed");
        var dog: Watchdog = .{ .io = f.io, .stream = conn.stream_reader.stream, .deadline = deadline };
        const thread = std.Thread.spawn(.{}, Watchdog.run, .{&dog}) catch null;
        defer {
            dog.done.store(true, .release);
            if (thread) |th| th.join();
        }
        const uri: std.Uri = .{ .scheme = if (t.https) "https" else "http", .host = .{ .raw = t.host }, .port = t.port, .path = .{ .percent_encoded = pathOf(t.target) }, .query = if (queryOf(t.target)) |q| .{ .percent_encoded = q } else null };
        var req = f.client.request(.GET, uri, .{
            .connection = conn,
            .keep_alive = false,
            .redirect_behavior = .unhandled,
            .headers = .{ .accept_encoding = .{ .override = "identity" }, .user_agent = .{ .override = user_agent } },
            .extra_headers = &.{.{ .name = "Accept", .value = "image/jpeg, image/png, image/webp, image/gif" }},
        }) catch return refuse(problem, "could not fetch the image URL: the request failed");
        defer req.deinit();
        req.sendBodiless() catch return refuse(problem, "could not fetch the image URL: the request failed");
        var redirect_buf: [8192]u8 = undefined;
        var resp = req.receiveHead(&redirect_buf) catch |e| return switch (e) {
            error.HttpContentEncodingUnsupported => refuse(problem, "compressed HTTP image responses are unsupported"),
            else => refuse(problem, if (dog.fired.load(.acquire)) "image download timed out" else "could not fetch the image URL: no valid HTTP reply"),
        };
        _ = try f.remaining(deadline, problem);
        const st = @intFromEnum(resp.head.status);
        if (st == 301 or st == 302 or st == 303 or st == 307 or st == 308) {
            const loc = resp.head.location orelse return refuse(problem, "image redirect has no destination");
            return .{ .location = try a.dupe(u8, loc) };
        }
        if (st != 200) return refuse(problem, try std.fmt.allocPrint(a, "image download returned HTTP {d}", .{st}));
        if (resp.head.content_encoding != .identity) return refuse(problem, "compressed HTTP image responses are unsupported");
        const ctype: []const u8 = resp.head.content_type orelse "";
        const media = std.mem.trim(u8, ctype[0 .. std.mem.indexOfScalar(u8, ctype, ';') orelse ctype.len], " \t");
        const known = for (media_types) |m| {
            if (std.ascii.eqlIgnoreCase(media, m)) break true;
        } else false;
        if (!known) return refuse(problem, "image URL content type must be JPEG, PNG, WebP or GIF");
        const too_big = try std.fmt.allocPrint(a, "image larger than {d} bytes (TF_DSV41_VISION_MAX_BYTES)", .{max_bytes});
        if (resp.head.content_length) |n| if (n > max_bytes) return refuse(problem, too_big);
        var tbuf: [64 * 1024]u8 = undefined;
        const body = resp.reader(&tbuf);
        const data = body.allocRemaining(a, .limited(max_bytes + 1)) catch |e| return switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            error.StreamTooLong => refuse(problem, too_big),
            else => refuse(problem, if (dog.fired.load(.acquire)) "image download timed out" else "could not fetch the image URL: the reply broke off"),
        };
        if (data.len > max_bytes) return refuse(problem, too_big);
        _ = try f.remaining(deadline, problem);
        return .{ .body = data };
    }
};

fn pathOf(target: []const u8) []const u8 {
    return target[0 .. std.mem.indexOfScalar(u8, target, '?') orelse target.len];
}
fn queryOf(target: []const u8) ?[]const u8 {
    const i = std.mem.indexOfScalar(u8, target, '?') orelse return null;
    return target[i + 1 ..];
}

/// ``urljoin(url, location)``: absolute, scheme-relative, absolute-path and relative references.
fn resolveRef(a: Allocator, t: Target, url: []const u8, loc: []const u8) Allocator.Error![]const u8 {
    if (std.mem.indexOf(u8, loc, "://") != null) return a.dupe(u8, loc);
    const scheme = if (t.https) "https" else "http";
    if (std.mem.startsWith(u8, loc, "//")) return std.fmt.allocPrint(a, "{s}:{s}", .{ scheme, loc });
    const after = url[std.mem.indexOf(u8, url, "://").? + 3 ..];
    const origin = url[0 .. url.len - after.len + (std.mem.indexOfAny(u8, after, "/?#") orelse after.len)];
    if (std.mem.startsWith(u8, loc, "/")) return std.fmt.allocPrint(a, "{s}{s}", .{ origin, loc });
    const path = pathOf(t.target);
    const dir = path[0 .. (std.mem.lastIndexOfScalar(u8, path, '/') orelse 0) + 1];
    return std.fmt.allocPrint(a, "{s}{s}{s}", .{ origin, dir, loc });
}

/// image_fetch's timer: the socket shut when the deadline passes with the request still running.
const Watchdog = struct {
    io: std.Io,
    stream: net.Stream,
    deadline: i96,
    done: std.atomic.Value(bool) = .init(false),
    fired: std.atomic.Value(bool) = .init(false),

    fn run(d: *Watchdog) void {
        while (!d.done.load(.acquire)) {
            if (std.Io.Clock.awake.now(d.io).nanoseconds >= d.deadline) {
                d.fired.store(true, .release);
                d.stream.shutdown(d.io, .both) catch {};
                return;
            }
            std.Io.sleep(d.io, .fromMilliseconds(10), .awake) catch {};
        }
    }
};

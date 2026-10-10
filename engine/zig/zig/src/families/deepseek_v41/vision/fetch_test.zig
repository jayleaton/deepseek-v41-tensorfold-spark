//! The image fetcher against prod's ``image_fetch.py`` (fixtures/fetch.jsonl from tools/zig/dsv41_vision/
//! gen_fetch_golden.py): every address's ``public_ip``, every URL's ``check_url`` under the four policies; then a
//! fetch from an in-process HTTP server (the local-testing policy): a redirect to an image, the declared type, the
//! size cap and an HTTP error.
const std = @import("std");
const testing = std.testing;
const fetch = @import("fetch.zig");
const parts = @import("parts.zig");
const net = std.Io.net;

const Rec = struct {
    ip: ?[]const u8 = null,
    ip4: ?[4]u8 = null,
    url4: ?[4]u8 = null,
    public: ?bool = null,
    url: ?[]const u8 = null,
    http: bool = false,
    private: bool = false,
    ok: bool = false,
    https: bool = false,
    host: []const u8 = "",
    port: u16 = 0,
    target: []const u8 = "",
    @"error": []const u8 = "",
};

test "public_ip and check_url decide as prod's image_fetch" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try std.Io.Dir.cwd().readFileAlloc(testing.io, "zig/src/families/deepseek_v41/vision/fixtures/fetch.jsonl", a, .limited(1 << 22));
    var it = std.mem.tokenizeScalar(u8, text, '\n');
    var n: usize = 0;
    var bad: usize = 0;
    while (it.next()) |line| {
        var r = try std.json.parseFromSliceLeaky(Rec, a, line, .{ .ignore_unknown_fields = true });
        if (r.ip4) |b| r.ip = try std.fmt.allocPrint(a, "{d}.{d}.{d}.{d}", .{ b[0], b[1], b[2], b[3] });
        if (r.url4) |b| {
            r.host = try std.fmt.allocPrint(a, "{d}.{d}.{d}.{d}", .{ b[0], b[1], b[2], b[3] });
            r.url = try std.mem.replaceOwned(u8, a, r.url.?, "{host}", r.host);
        }
        n += 1;
        if (r.ip) |ip| {
            const addr = try parseIp(ip);
            if (fetch.publicIp(addr) != r.public.?) {
                std.debug.print("public_ip {s}: {} (Python {})\n", .{ ip, !r.public.?, r.public.? });
                bad += 1;
            }
            continue;
        }
        var p: parts.Problem = .{};
        const got = fetch.checkUrl(a, r.url.?, .{ .http = r.http, .private = r.private }, &p);
        if (got) |t| {
            if (!r.ok or t.https != r.https or t.port != r.port or !std.mem.eql(u8, t.host, r.host) or !std.mem.eql(u8, t.target, r.target)) {
                std.debug.print("check_url {s} (http {} private {}): {s} {d} {s}, Python ok {} {s} {d} {s} {s}\n", .{ r.url.?, r.http, r.private, t.host, t.port, t.target, r.ok, r.host, r.port, r.target, r.@"error" });
                bad += 1;
            }
        } else |_| if (r.ok or !std.mem.eql(u8, p.message, r.@"error")) {
            std.debug.print("check_url {s} (http {} private {}): refused \"{s}\", Python ok {} \"{s}\"\n", .{ r.url.?, r.http, r.private, p.message, r.ok, r.@"error" });
            bad += 1;
        }
    }
    std.debug.print("fetch golden: {d}/{d} decisions as image_fetch.py\n", .{ n - bad, n });
    try testing.expectEqual(@as(usize, 0), bad);
}

/// Any textual address Python prints (RFC 4291 forms with "::" and an embedded IPv4 tail; std's parser takes fewer).
fn parseIp(text: []const u8) !net.IpAddress {
    if (std.mem.indexOfScalar(u8, text, ':') == null) return net.IpAddress.parse(text, 0);
    var groups: [8]u16 = @splat(0);
    const dbl = std.mem.indexOf(u8, text, "::");
    const head = if (dbl) |d| text[0..d] else text;
    const tail = if (dbl) |d| text[d + 2 ..] else "";
    var hg: [8]u16 = undefined;
    var tg: [8]u16 = undefined;
    const nh = try groupsOf(head, &hg);
    const nt = try groupsOf(tail, &tg);
    @memcpy(groups[0..nh], hg[0..nh]);
    @memcpy(groups[8 - nt ..], tg[0..nt]);
    var bytes: [16]u8 = undefined;
    for (groups, 0..) |g, i| std.mem.writeInt(u16, bytes[2 * i ..][0..2], g, .big);
    return .{ .ip6 = .{ .port = 0, .bytes = bytes } };
}

fn groupsOf(s: []const u8, out: *[8]u16) !usize {
    if (s.len == 0) return 0;
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, s, ':');
    while (it.next()) |g| {
        if (std.mem.indexOfScalar(u8, g, '.') != null) {
            const v4 = (try net.IpAddress.parse(g, 0)).ip4.bytes;
            out[n] = @as(u16, v4[0]) << 8 | v4[1];
            out[n + 1] = @as(u16, v4[2]) << 8 | v4[3];
            n += 2;
        } else {
            out[n] = try std.fmt.parseInt(u16, g, 16);
            n += 1;
        }
    }
    return n;
}

/// A one-thread HTTP server: each accepted connection gets the next canned reply.
const Server = struct {
    io: std.Io,
    srv: net.Server,
    replies: []const []const u8,

    fn run(s: *Server) void {
        for (s.replies) |reply| {
            var stream = s.srv.accept(s.io) catch return;
            defer stream.close(s.io);
            var rb: [4096]u8 = undefined;
            var r = stream.reader(s.io, &rb);
            // the request head, to its blank line
            while (true) {
                const line = r.interface.takeDelimiterExclusive('\n') catch break;
                r.interface.toss(1);
                if (std.mem.trim(u8, line, "\r").len == 0) break;
            }
            var wb: [4096]u8 = undefined;
            var w = stream.writer(s.io, &wb);
            w.interface.writeAll(reply) catch {};
            w.interface.flush() catch {};
        }
    }
};

test "a fetch follows a checked redirect, takes a declared image, refuses the wrong type, the oversize and HTTP errors" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = testing.io;
    var addr: net.IpAddress = .{ .ip4 = .{ .bytes = .{ 127, 0, 0, 1 }, .port = 0 } };
    var srv = try addr.listen(io, .{ .reuse_address = true });
    const port = srv.socket.address.getPort();
    const loc = try std.fmt.allocPrint(a, "HTTP/1.1 302 Found\r\nLocation: /img.png\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", .{});
    const img = "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: 4\r\nConnection: close\r\n\r\nPNG!";
    const html = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 2\r\nConnection: close\r\n\r\nhi";
    const big = "HTTP/1.1 200 OK\r\nContent-Type: image/gif\r\nContent-Length: 9\r\nConnection: close\r\n\r\n123456789";
    const notfound = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
    var server: Server = .{ .io = io, .srv = srv, .replies = &.{ loc, img, html, big, notfound } };
    const th = try std.Thread.spawn(.{}, Server.run, .{&server});
    defer {
        th.join();
        srv.deinit(io);
    }
    var f: fetch.Fetcher = .{ .gpa = testing.allocator, .io = io, .policy = .{ .http = true, .private = true }, .client = .{ .allocator = testing.allocator, .io = io } };
    defer f.deinit();
    const url = try std.fmt.allocPrint(a, "http://localhost:{d}/start", .{port});
    const deadline = std.Io.Clock.awake.now(io).nanoseconds + 10 * std.time.ns_per_s;
    var p: parts.Problem = .{};
    try testing.expectEqualStrings("PNG!", try f.fetch(a, url, 1024, deadline, &p));
    try testing.expectError(error.Refused, f.fetch(a, url, 1024, deadline, &p));
    try testing.expectEqualStrings("image URL content type must be JPEG, PNG, WebP or GIF", p.message);
    try testing.expectError(error.Refused, f.fetch(a, url, 8, deadline, &p));
    try testing.expectEqualStrings("image larger than 8 bytes (TF_DSV41_VISION_MAX_BYTES)", p.message);
    try testing.expectError(error.Refused, f.fetch(a, url, 1024, deadline, &p));
    try testing.expectEqualStrings("image download returned HTTP 404", p.message);
    // the default policy refuses this URL before any connection: http, a private address
    var strict: fetch.Fetcher = .{ .gpa = testing.allocator, .io = io, .policy = .{}, .client = .{ .allocator = testing.allocator, .io = io } };
    defer strict.deinit();
    try testing.expectError(error.Refused, strict.fetch(a, url, 1024, deadline, &p));
    try testing.expectEqualStrings("image URL must be HTTPS on port 443, without credentials or a fragment", p.message);
}

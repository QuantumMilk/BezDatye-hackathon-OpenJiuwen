"""Linux worker. Apply OS restrictions before loading task code."""

import ctypes
import errno
import os
import platform
import resource
import sys
import sysconfig

UNAVAILABLE = 125
READ_FILE = 1 << 2
READ_DIR = 1 << 3
HANDLED_FS = (1 << 15) - 1  # Landlock ABI 3, including truncate.


class Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def restrict_files(workspace):
    if platform.machine() not in {"x86_64", "aarch64"}:
        raise RuntimeError("Unknown Landlock syscall numbers")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.syscall(444, 0, 0, 1) < 3:
        raise RuntimeError("Landlock ABI 3 is unavailable")
    attributes = Ruleset(HANDLED_FS)
    ruleset = libc.syscall(444, ctypes.byref(attributes), ctypes.sizeof(attributes), 0)
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    try:
        for directory in {workspace, sysconfig.get_path("stdlib"), "/usr/lib"}:
            fd = os.open(directory, os.O_PATH | os.O_DIRECTORY)
            try:
                rule = PathRule(READ_FILE | READ_DIR, fd)
                if libc.syscall(445, ruleset, 1, ctypes.byref(rule), 0) < 0:
                    raise OSError(ctypes.get_errno(), "landlock_add_rule")
            finally:
                os.close(fd)
        if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.syscall(446, ruleset, 0) < 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    finally:
        os.close(ruleset)


def restrict_syscalls():
    library = ctypes.CDLL("libseccomp.so.2")
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_rule_add.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint,
    ]
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    context = library.seccomp_init(0x00050000 | errno.EPERM)
    if not context:
        raise RuntimeError("Could not initialize seccomp")
    try:
        # No sockets, process creation, exec, signals to other processes,
        # ptrace, pidfds or io_uring. Filesystem writes are denied by Landlock.
        allowed = """
            read write readv writev pread64 close lseek fstat newfstatat statx
            stat lstat getdents64 openat access faccessat faccessat2 readlink
            readlinkat getcwd fcntl ioctl dup dup2 dup3 mmap mprotect munmap
            mremap madvise brk futex clock_gettime clock_nanosleep nanosleep
            gettimeofday getpid gettid getuid geteuid getgid getegid getrandom
            rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack sched_yield
            sched_getaffinity getrusage times uname exit exit_group
        """
        for name in allowed.split():
            number = library.seccomp_syscall_resolve_name(name.encode("ascii"))
            if number >= 0 and library.seccomp_rule_add(context, 0x7FFF0000, number, 0) != 0:
                raise RuntimeError(f"Could not add seccomp rule for {name}")
        if library.seccomp_load(context) != 0:
            raise RuntimeError("Could not load seccomp")
    finally:
        library.seccomp_release(context)


def main():
    workspace, mode = sys.argv[1:]
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
        restrict_files(workspace)
        restrict_syscalls()
    except (OSError, RuntimeError, ValueError):
        return UNAVAILABLE

    sys.path.insert(0, workspace)
    if mode == "probe":
        return 0
    if mode == "tests":
        import runpy

        sys.argv = ["test_mylib.py"]
        runpy.run_path("test_mylib.py", run_name="__main__")
    else:
        source = sys.stdin.buffer.read(65537)
        if len(source) > 65536:
            return 2
        exec(compile(source, "<task>", "exec"), {"__name__": "__main__"})
    return 0


if __name__ == "__main__":
    sys.exit(main())

# C++ Thread Creation and Join Flow

```mermaid
sequenceDiagram
    participant Main as Main Thread
    participant std as std::thread API
    participant libpthread as libpthread
    participant kernel as Linux Kernel

    Main->>std: std::thread t(func)
    std->>libpthread: pthread_create()
    libpthread->>kernel: clone()
    kernel-->>libpthread: new thread created
    libpthread-->>std: return success
    std-->>Main: thread object created
    Main->>std: t.join()
    std->>libpthread: pthread_join()
    libpthread->>kernel: wait for thread exit
    kernel-->>libpthread: thread exited
    libpthread-->>std: cleanup
    std-->>Main: join returns

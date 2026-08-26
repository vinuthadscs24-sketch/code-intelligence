
package com.example.audit;

interface Repository {
    void save();
}

public class OuterRepository implements Repository {
    private Helper helper = new Helper();

    public void save() {
        this.helper.log();
        helper.flush();
    }

    public void save(String name) {
        OuterRepository.staticAudit(name);
    }

    public void save(String name, boolean flush) {
        AuditUtils.create();
    }

    public static void staticAudit(String str) {}

    public class InnerLogger {
        public void save() {
            System.out.println("Inner save");
        }
    }
}

class Helper {
    public void log() {}
    public void flush() {}
}

class AuditUtils {
    public static void create() {}
}

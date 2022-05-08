package org.apache.hadoop.yarn.server.resourcemanager.scheduler.fair;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.yarn.webapp.hamlet.HamletSpec;

public class FlowScheduler extends FairScheduler {
    private static final Log LOG = LogFactory.getLog(FlowScheduler.class);
    @Override
    public void update() {
        System.out.println("FlowScheduler.Update()");
        LOG.info("FlowScheduler.Update()");
        super.update();
    }

    @Override
    public FSContext getContext() {
        System.out.println("FlowScheduler.getContext()");
        LOG.info("FlowScheduler.getContext()");
        return super.getContext();
    }
}

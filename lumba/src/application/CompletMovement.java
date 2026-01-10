package application;

import javax.inject.Inject;

import com.kuka.roboticsAPI.applicationModel.RoboticsAPIApplication;
import static com.kuka.roboticsAPI.motionModel.BasicMotions.*;

import com.kuka.roboticsAPI.deviceModel.LBR;
import com.kuka.roboticsAPI.geometricModel.Frame;
import com.kuka.roboticsAPI.motionModel.LIN;

public class CompletMovement extends RoboticsAPIApplication {
    @Inject
    private LBR lbr;

    private static final double SPEED_MM_S = 0.25;
    private static final double DELTA_Z_MM = 10.0;

    public void initialize() {
    }

    public void run() {
        Frame start = lbr.getCurrentCartesianPosition(lbr.getFlange());
        Frame down = start.copy();
        down.setZ(down.getZ() - DELTA_Z_MM);

        LIN downMotion = lin(down);
        downMotion.setCartVelocity(SPEED_MM_S);
        lbr.move(downMotion);

        Frame up = start.copy();
        up.setZ(start.getZ());

        LIN upMotion = lin(up);
        upMotion.setCartVelocity(SPEED_MM_S);
        lbr.move(upMotion);
    }
}

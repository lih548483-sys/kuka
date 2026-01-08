package application;

import javax.inject.Inject;

import com.kuka.roboticsAPI.applicationModel.RoboticsAPIApplication;
import static com.kuka.roboticsAPI.motionModel.BasicMotions.*;

import com.kuka.roboticsAPI.deviceModel.LBR;
import com.kuka.roboticsAPI.geometricModel.Frame;
import com.kuka.roboticsAPI.motionModel.LIN;

public class MoveZConstantSpeed extends RoboticsAPIApplication {
    @Inject
    private LBR lbr;

    private static final double SPEED_MM_S = 0.25;
    private static final double DELTA_Z_MM = 10.0;

    public void initialize() {
    }

    public void run() {
        Frame start = lbr.getCurrentCartesianPosition(lbr.getFlange());
        Frame target = start.copy();
        target.setZ(target.getZ() - DELTA_Z_MM);

        LIN linMotion = lin(target);
        linMotion.setCartVelocity(SPEED_MM_S);
        lbr.move(linMotion);
    }
}

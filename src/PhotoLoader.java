import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;

/**
 * Created by Peter on 2015/3/3.
 */
public class PhotoLoader {
    public static int[][][] loadFile(String filename){
        BufferedImage img = null;
        try {
            img = ImageIO.read(new File(filename));
        } catch (IOException e) {
            e.printStackTrace();
        }
        int width = img.getWidth();
        int height = img.getHeight();
        int[][][] mat;
        mat = getArrayFromImage(img);
        return mat;
    }

    public static void saveFile(int width, int height, int[][][] mat, String filename){
        int[] output_vec = new int[3*width*height];
        int c = 0;
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                for (int index = 0; index < 3; index++) {
                    output_vec[c] = mat[index][row][col];
                    c++;
                }
            }
        }
        BufferedImage img = null;
        img = getImageFromArray(output_vec, width, height);
        try {
            ImageIO.write(img, "jpg", new File(filename));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public static int[][][] getArrayFromImage(BufferedImage image){
        final byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        final int width = image.getWidth();
        final int height = image.getHeight();
        int[][][] mat = new int[3][height][width];
        for (int pixel = 0, row = 0, col = 0; pixel< pixels.length; pixel += 3){
            mat[2][row][col] = ((int) pixels[pixel] & 0xff); // blue
            mat[1][row][col]= ((int) pixels[pixel + 1] & 0xff); // green
            mat[0][row][col]= ((int) pixels[pixel + 2] & 0xff);  // red
            col++;
            if (col == width){
                col = 0;
                row++;
            }
        }
        return mat;
    }

    public static BufferedImage getImageFromArray(int[] pixels, int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        WritableRaster raster = (WritableRaster) image.getData();
        raster.setPixels(0,0,width,height,pixels);
        image.setData(raster);
        return image;
    }

    public static void main(String[] args) {
        int[][][] mat;
        mat = loadFile("1.jpg");
        int height = mat[0].length;
        int width = mat[0][0].length;

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                for (int index = 0; index < 3; index++) {
                    mat[index][row][col] = mat[2][row][col];
                }
            }
        }

        saveFile(width, height, mat, "result.jpg");
    }
}

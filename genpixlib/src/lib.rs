#![feature(plugin)]
#![cfg_attr(test, plugin(stainless))]

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(non_snake_case)]
extern crate image;

use std::fs::File;
use std::path::Path;
use std::collections::HashMap;

use image::{GenericImage, Pixel, ImageBuffer};

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Data {
    RGBA(u8, u8, u8, u8),
    Empty,
}

impl Default for Data {
    fn default() -> Data {
        Data::Empty
    }
}
enum Flip {
    Horizontally,
    Vertically,
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Point {
    x: i64,
    y: i64,
}

#[derive(Debug)]
pub struct Bounds {
    min: Point,
    max: Point,
}

pub trait OutputPx {
    fn get_pixel_at(&self, x: u32, y: u32) -> Data;
}
impl Bounds {
    pub fn dimensions(&self) -> (i64, i64) {
        let width = self.max.x - self.min.x;
        let height = self.max.y - self.min.y;
        (width + 1, height + 1)
    }

    pub fn from_points<'a, T>(points: T) -> Bounds
        where T: Iterator<Item = &'a Point>
    {
        let mut xmin = 0;
        let mut ymin = 0;
        let mut xmax = 0;
        let mut ymax = 0;
        let mut initialized = false;

        for point in points {
            if initialized {
                if point.x < xmin {
                    xmin = point.x
                }
                if point.x > xmax {
                    xmax = point.x
                }
                if point.y < ymin {
                    ymin = point.y
                }
                if point.y > ymax {
                    ymax = point.y
                }
            } else {
                xmin = point.x;
                ymin = point.y;
                xmax = point.x;
                ymax = point.y;
                initialized = true;
            }
        }

        let min = Point { x: xmin, y: ymin };
        let max = Point { x: xmax, y: ymax };

        Bounds {
            min: min,
            max: max,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Grid {
    plane: HashMap<(i64, i64), Data>,
}

impl Grid {
    pub fn new() -> Grid {
        Grid { plane: HashMap::new() }
    }

    pub fn get(&self, x: i64, y: i64) -> Option<&Data> {
        self.plane.get(&(x, y))
    }

    pub fn set(&mut self, x: i64, y: i64, data: Data) -> Option<Data> {
        self.plane.insert((x, y), data)
    }

    pub fn bounds(&self) -> Bounds {
        let points = self.plane.keys().map(|p| Point { x: p.0, y: p.1 }).collect::<Vec<Point>>();
        Bounds::from_points(points.iter())
    }

    pub fn size(&self) -> (i64, i64) {
        self.bounds().dimensions()
    }

    pub fn translate_by(&mut self, x: i64, y: i64) {
        let mut new_plane = HashMap::new();
        for (point, data) in &self.plane {
            let x_new = point.0 + x;
            let y_new = point.1 + y;
            new_plane.insert((x_new, y_new), *data);
        }
        self.plane = new_plane;
    }

    pub fn translate_to(&mut self, x: i64, y: i64) {
        let bounds = self.bounds();
        let tx = x - bounds.min.x;
        let ty = y - bounds.min.y;
        self.translate_by(tx, ty);
    }

    pub fn merge_at(&mut self, grid: &Grid, x: i64, y: i64) {
        let mut translated_grid = grid.clone();
        translated_grid.translate_to(x, y);
        self.merge(&translated_grid);
    }

    pub fn merge(&mut self, grid: &Grid) {
        for (point, data) in grid.plane.iter() {
            self.plane.insert(*point, *data);
        }
    }

    pub fn clear(&mut self) {
        self.plane = HashMap::new();
    }

    pub fn replace_all(&mut self, new_grid: Grid) {
        self.plane = new_grid.plane;
    }

    fn do_flip(&mut self, flip: Flip) {
        let mut new_plane: HashMap<(i64, i64), Data> = HashMap::new();
        for (coords, data) in self.plane.iter() {
            match flip {
                Flip::Horizontally => {
                    new_plane.insert((-coords.0, coords.1), *data);
                }
                Flip::Vertically => {
                    new_plane.insert((coords.0, -coords.1), *data);
                }
            };
        }
        self.plane = new_plane;
    }

    pub fn flip_horizontally(&mut self) {
        self.do_flip(Flip::Horizontally);
    }

    pub fn flip_vertically(&mut self) {
        self.do_flip(Flip::Vertically);
    }
}

impl OutputPx for Grid {
    fn get_pixel_at(&self, x: u32, y: u32) -> Data {
        match self.get(x as i64, y as i64) {
            Some(data) => return *data,
            None => return Data::RGBA(0, 0, 0, 255),
        }
    }
}

mod CellTool {

    use super::Grid;
    pub fn offset(source: (i64, i64), target: (i64, i64)) -> (i64, i64) {
        let (xs, ys) = (source.0, source.1);
        let (xt, yt) = (target.0, target.1);

        let x_offset = xt - xs;
        let y_offset = yt - ys;

        (x_offset, y_offset)
    }

    pub fn distance(source: (i64, i64), target: (i64, i64)) -> f32 {
        let (xs, ys) = (source.0 as f32, source.1 as f32);
        let (xt, yt) = (target.0 as f32, target.1 as f32);

        ((xt - xs).powi(2) + (yt - ys).powi(2)).sqrt()
    }

    pub fn is_adjacent(source: (i64, i64), target: (i64, i64)) -> bool {
        let (xs, ys) = (source.0, source.1);
        let (xt, yt) = (target.0, target.1);

        let x_distance = (xs - xt).abs();
        let y_distance = (ys - yt).abs();

        if x_distance > 1 || y_distance > 1 {
            return false;
        }

        if x_distance == 1 && y_distance == 1 {
            return false;
        }

        return true;
    }

    pub fn get_adjacent_coords(target: (i64, i64)) -> Vec<(i64, i64)> {
        let mut coords: Vec<(i64, i64)> = Vec::new();
        coords.push((target.0 - 1, target.1));
        coords.push((target.0 + 1, target.1));
        coords.push((target.0, target.1 - 1));
        coords.push((target.0, target.1 + 1));
        coords
    }

    pub fn get_corner_coords(target: (i64, i64)) -> Vec<(i64, i64)> {
        let mut coords: Vec<(i64, i64)> = Vec::new();
        coords.push((target.0 - 1, target.1 - 1));
        coords.push((target.0 - 1, target.1 + 1));
        coords.push((target.0 + 1, target.1 - 1));
        coords.push((target.0 + 1, target.1 + 1));
        coords
    }

    pub fn get_surrounding_coords(target: (i64, i64)) -> Vec<(i64, i64)> {
        let mut surrounding = Vec::new();
        let mut corners = get_corner_coords(target);
        let mut adjacent = get_adjacent_coords(target);
        surrounding.append(&mut corners);
        surrounding.append(&mut adjacent);
        surrounding
    }

    #[cfg(test)]
    mod tests {
        use super::{get_adjacent_coords, is_adjacent, get_surrounding_coords, get_corner_coords,
                    offset, distance, expand_cells};
        use super::super::{Grid, Data, Point, Bounds};
        describe! celltool {

            it "determines if a cell is adjacent" {
                let source = (1,1);
                let adjacent = (1,2);
                assert_eq!(is_adjacent(source, adjacent), true);
            }

            it "determines if a non-adjacent cell is non-adjacent" {
                let source = (1,1);
                let nonadjacent = (2,2);
                assert_eq!(is_adjacent(source, nonadjacent), false);
            }

            it "gets adjacent cell coordinates" {
                let target = (0,0);
                let coords = vec![(-1,0),(1,0),(0,-1),(0,1)];
                let adjacent = get_adjacent_coords(target);
                assert_eq!(adjacent, coords);
            }

            it "gets corner cell coordinates" {
                let target = (0,0);
                let coords = vec![(-1,-1),(-1,1),(1,-1),(1,1)];
                let corners = get_corner_coords(target);
                assert_eq!(corners, coords);
            }

            it "gets surrounding cell coordinates" {
                let target = (0,0);
                let coords = vec![(-1,-1),(-1,1),(1,-1),(1,1),(-1,0),(1,0),(0,-1),(0,1)];
                let surrounding = get_surrounding_coords(target);
                assert_eq!(surrounding, coords);
            }

            it "calculates negative offsets" {
                let source = (0,0);
                let target = (-2,-2);
                let offset = offset(source, target);
                assert_eq!(offset.0, -2);
                assert_eq!(offset.1, -2);
            }

            it "calculates positive offsets" {
                let source = (0,0);
                let target = (2,2);
                let offset = offset(source, target);
                assert_eq!(offset.0, 2);
                assert_eq!(offset.1, 2);
            }

            it "calculates distance" {
                let source = (0,0);
                let target = (3,2);
                let d = distance(source, target);
                assert_eq!(d, 13_f32.sqrt());
            }
        }
    }

}

mod GridTool {
    use super::{Grid, Data};

    #[cfg(test)]
    mod tests {
        use super::{Grid, Data};

        describe! gridtool {
            before_each {
                let grid = Grid::new();
            }
            it "does nothing" {

            }
        }

    }
}

pub mod ImageTool {
    use std::fs::File;
    use std::io;
    use std::path::Path;
    use std::collections::HashMap;

    extern crate image;
    use image::{GenericImage, Pixel, ImageBuffer, Primitive};

    use super::{OutputPx, Data, Point, Grid};

    pub trait GridExporter {
        fn from_grid<T>(pixgrid: &T,
                        src_size: (u32, u32))
                        -> ImageBuffer<image::Rgba<u8>, Vec<u8>>
            where T: OutputPx;
    }

    impl GridExporter for ImageBuffer<image::Rgba<u8>, Vec<u8>> {
        fn from_grid<T>(pixgrid: &T, src_size: (u32, u32)) -> ImageBuffer<image::Rgba<u8>, Vec<u8>>
            where T: OutputPx
        {
            let (imgx, imgy) = src_size;
            let mut imgbuf = image::ImageBuffer::new(imgx, imgy);
            let mut x = 0;
            let mut y = 0;
            loop {
                match pixgrid.get_pixel_at(x, y) {
                    Data::RGBA(r, g, b, a) => {
                        let px = image::Rgba([r, g, b, a]);
                        let y_coord = imgy - y - 1;
                        imgbuf.put_pixel(x as u32, y_coord as u32, px);
                    }
                    _ => (),
                }
                x += 1;
                if x == imgx {
                    x = 0;
                    y += 1;
                    if y == imgy {
                        break;
                    }
                }
            }
            imgbuf
        }
    }

    pub fn new_file(path: &str) -> Result<File, io::Error> {
        File::create(&Path::new(path))
    }

    pub fn save_png(mut fh: File,
                    imgbuf: ImageBuffer<image::Rgba<u8>, Vec<u8>>)
                    -> Result<(), image::ImageError> {
        let ref mut fh = fh;
        image::ImageRgba8(imgbuf).save(fh, image::PNG)
    }

    pub fn save_grid(mut fh: File,
                     grid: &Grid,
                     width: u32,
                     height: u32)
                     -> Result<(), image::ImageError> {
        let ref mut fh = fh;
        let buf = ImageBuffer::from_grid(grid, (width, height));
        image::ImageRgba8(buf).save(fh, image::PNG)
    }

    #[cfg(test)]
    mod tests {
        use ImageTool;
        use image::ImageBuffer;
        use super::GridExporter;
        use super::super::{Grid, Data, Point, Bounds};

        describe! imagetool {
            before_each {
                let grid = Grid::new();
            }

            it "converts a Grid to an image buffer" {
                let mut grid = grid;
                let white = Data::RGBA(255,255,255,255);
                let red = Data::RGBA(255,0,0,255);
                grid.set(0,0,red);
                grid.set(1,1,white);
                grid.set(2,2,red);
                let size = grid.bounds().dimensions();
                let buf = ImageBuffer::from_grid(&grid, (size.0 as u32, size.1 as u32));
                let pix_00 = buf.get_pixel(0,2);
                let pix_11 = buf.get_pixel(1,1);
                let pix_22 = buf.get_pixel(2,0);
                assert_eq!(pix_00.data[0], 255);
                assert_eq!(pix_00.data[1], 0);
                assert_eq!(pix_00.data[2], 0);

                assert_eq!(pix_11.data[0], 255);
                assert_eq!(pix_11.data[1], 255);
                assert_eq!(pix_11.data[2], 255);

                assert_eq!(pix_22.data[0], 255);
                assert_eq!(pix_22.data[1], 0);
                assert_eq!(pix_22.data[2], 0);
            }
        }
    }
}

pub mod Gen {
    extern crate rand;
    use Gen::rand::{Rng, SeedableRng, StdRng, ThreadRng, thread_rng};
    use Gen::rand::distributions::{Range, IndependentSample};
    use super::{Grid, Data, CellTool};

    pub struct Generator {
        seed_gen: ThreadRng,
        rng: StdRng,
    }

    impl Generator {
        pub fn new() -> Generator {
            let mut seed_gen = thread_rng();
            let seed: &[_] = &[seed_gen.gen(), seed_gen.gen(), seed_gen.gen()];
            let rng: StdRng = StdRng::from_seed(seed);
            Generator {
                seed_gen: seed_gen,
                rng: rng,
            }
        }
        pub fn new_seed<'a>(&'a mut self) -> Vec<usize> {
            vec![self.seed_gen.gen(), self.seed_gen.gen(), self.seed_gen.gen()]
        }

        pub fn reseed(&mut self) {
            let seed = self.new_seed();
            self.rng.reseed(seed.as_slice());
        }

        pub fn invader_seeded(&mut self,
                              seed: Vec<usize>,
                              width: u32,
                              height: u32,
                              min_px: u32,
                              max_px: u32,
                              max_nearby: (u32, u32),
                              max_edge: (u32, u32),
                              center_overlap: i32)
                              -> Option<Grid> {
            self.rng.reseed(seed.as_slice());

            let mut max_px = max_px;
            if min_px >= max_px {
                max_px = min_px + 1;
            }
            let total_pixels = Range::new(min_px, max_px + 1).ind_sample(&mut self.rng);
            let max_nearby = Range::new(max_nearby.0, max_nearby.1 + 1).ind_sample(&mut self.rng);
            let max_edge = Range::new(max_edge.0, max_edge.1 + 1).ind_sample(&mut self.rng);
            let x_rng = Range::new(0, width);
            let y_rng = Range::new(0, height + 1);
            let mut grid = Grid::new();

            let mut pixels_filled = 0;

            let mut edge_pixels = 0;

            println!("px: {}:{}->{}, nearby: {}, edge: {}",
                     min_px,
                     max_px,
                     total_pixels,
                     max_nearby,
                     max_edge);

            let mut iterations = 0;

            while pixels_filled < total_pixels {
                iterations += 1;
                if iterations > 10000 {
                    return None;
                }
                let x = x_rng.ind_sample(&mut self.rng);
                let y = y_rng.ind_sample(&mut self.rng);

                if grid.get(x as i64, y as i64).is_some() {
                    continue;
                }

                if x == width - 1 && edge_pixels >= max_edge {
                    continue;
                }

                if x == width - 1 {
                    edge_pixels += 1;
                }

                let nearby_pixels = CellTool::get_surrounding_coords((x as i64, y as i64));
                let mut num_nearby = 0;
                for coords in nearby_pixels {
                    if grid.get(coords.0 as i64, coords.1 as i64).is_some() {
                        num_nearby += 1;
                    }
                }

                if pixels_filled == 0 || (num_nearby > 0 && num_nearby <= max_nearby) {
                    let pixel = Data::RGBA(255, 255, 255, 255);
                    if pixels_filled == 0 {
                        grid.set((width - 1) as i64, y as i64, pixel);
                    } else {
                        grid.set(x as i64, y as i64, pixel);
                    }
                    pixels_filled += 1;
                }
            }

            let mut dup_grid = grid.clone();
            dup_grid.flip_horizontally();
            dup_grid.translate_by((width as i64 * 2) - 1 - center_overlap as i64, 0);
            grid.merge(&dup_grid);

            if center_overlap != 0 {
                grid.translate_by((-center_overlap / 2) as i64, 0);
            }
            Some(grid)
        }

        pub fn invader(&mut self,
                       width: u32,
                       height: u32,
                       min_px: u32,
                       max_px: u32,
                       max_nearby: (u32, u32),
                       max_edge: (u32, u32),
                       center_overlap: i32)
                       -> Option<Grid> {
            let seed = self.new_seed();
            println!("Gen invader with seed :{:?}", seed);
            self.invader_seeded(seed,
                                width,
                                height,
                                min_px,
                                max_px,
                                max_nearby,
                                max_edge,
                                center_overlap)
        }
    }

    #[cfg(test)]
    mod tests {
        describe! generate {
            it "need generator tests" {
                assert!(false, "need tests to ensure basic generation");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Grid, Data, Point, Bounds};

    describe! grid {
        before_each {
            let grid = Grid::new();
        }

        it "adds and reads data" {
            let mut grid = grid;
            let data = Data::RGBA(1,1,1,1);
            grid.set(1,2,data);
            let read_data = grid.get(1,2).unwrap();
            assert_eq!(*read_data, Data::RGBA(1,1,1,1));
        }

        it "properly handles getting invalid data" {
            let read_data = grid.get(9,9);
            assert_eq!(read_data.is_some(), false);
        }

        it "reports proper bounding box" {
            let mut grid = grid;
            let data1: Data = Default::default();
            let data2: Data = Default::default();
            grid.set(1,1,data1);
            grid.set(2,2,data2);
            let bounds = grid.bounds();
            assert_eq!(bounds.min.x, 1);
            assert_eq!(bounds.min.y, 1);
            assert_eq!(bounds.max.x, 2);
            assert_eq!(bounds.max.y, 2);
        }

        it "reports proper dimensions" {
            let mut grid = grid;
            let data1: Data = Default::default();
            let data2: Data = Default::default();
            grid.set(1,1,data1);
            grid.set(2,3,data2);
            let dimensions = grid.size();
            assert_eq!(dimensions.0, 2);
            assert_eq!(dimensions.1, 3);
        }

        it "translates by coordinates" {
            let mut grid = grid;
            let data1 = Data::RGBA(1,1,1,1);
            let data2 = Data::RGBA(2,2,2,2);
            grid.set(1,1,data1);
            grid.set(2,3,data2);
            grid.translate_by(1, -1);
            let point1 = grid.get(2, 0).unwrap();
            let point2 = grid.get(3, 2).unwrap();
            assert_eq!(*point1, data1);
            assert_eq!(*point2, data2);
        }

        it "translates to coordinates" {
            let mut grid = grid;
            let data1 = Data::RGBA(1,1,1,1);
            let data2 = Data::RGBA(2,2,2,2);
            grid.set(1,1,data1);
            grid.set(2,3,data2);
            grid.translate_to(0, 0);
            let point1 = grid.get(0, 0).unwrap();
            let point2 = grid.get(1, 2).unwrap();
            assert_eq!(*point1, data1);
            assert_eq!(*point2, data2);
        }

        it "horizontally flips the grid" {
            let mut grid = grid;
            let data1 = Data::RGBA(1,1,1,1);
            let data2 = Data::RGBA(2,2,2,2);
            let data3 = Data::RGBA(4,4,4,4);
            grid.set(-1,-1,data1);
            grid.set(3,2,data2);
            grid.set(4,3,data3);
            grid.flip_horizontally();
            let point1 = grid.get(1, -1).unwrap();
            let point2 = grid.get(-3, 2).unwrap();
            let point3 = grid.get(-4, 3).unwrap();
            assert_eq!(*point1, data1);
            assert_eq!(*point2, data2);
            assert_eq!(*point3, data3);
        }

        it "vertically flips the grid" {
            let mut grid = grid;
            let data1 = Data::RGBA(1,1,1,1);
            let data2 = Data::RGBA(2,2,2,2);
            let data3 = Data::RGBA(4,4,4,4);
            grid.set(-1,-1,data1);
            grid.set(3,2,data2);
            grid.set(4,3,data3);
            grid.flip_vertically();
            let point1 = grid.get(-1, 1).unwrap();
            let point2 = grid.get(3, -2).unwrap();
            let point3 = grid.get(4, -3).unwrap();
            assert_eq!(*point1, data1);
            assert_eq!(*point2, data2);
            assert_eq!(*point3, data3);
        }

        it "merges another grid" {
            let mut grid = grid;
            let mut add_grid = Grid::new();
            let data1 = Data::RGBA(1,1,1,1);
            let data2 = Data::RGBA(2,2,2,2);

            add_grid.set(1, 1, data1);
            grid.set(2,2, data2);

            grid.merge(&add_grid);

            let point1 = grid.get(1,1).unwrap();
            let point2 = grid.get(2,2).unwrap();
            assert_eq!(*point1, data1);
            assert_eq!(*point2, data2);
        }

        it "merges another grid at a coordinate" {
            let mut grid = grid;
            let mut add_grid = Grid::new();
            let data1 = Data::RGBA(1,1,1,1);
            let data2 = Data::RGBA(2,2,2,2);

            add_grid.set(1, 1, data1);
            grid.set(2,2, data2);

            grid.merge_at(&add_grid, 0, 0);

            let point1 = grid.get(0,0).unwrap();
            let point2 = grid.get(2,2).unwrap();
            assert_eq!(*point1, data1);
            assert_eq!(*point2, data2);
        }

    }

    describe! bounds {
        before_each {
            let grid = Grid::new();
        }

        it "calculates dimensions" {
            let min = Point { x:0, y:0 };
            let max = Point { x:2, y:1 };
            let bounds = Bounds { min: min, max: max };
            let dimensions = bounds.dimensions();
            assert_eq!(dimensions.0, 3);
            assert_eq!(dimensions.1, 2);
        }

        it "determines bounding points from a collection" {
            let min = Point { x:0, y:0 };
            let max = Point { x:2, y:1 };
            let points = vec![max, min];
            let bounds = Bounds::from_points(points.iter());
            assert_eq!(bounds.min.x, 0);
            assert_eq!(bounds.min.y, 0);
            assert_eq!(bounds.max.x, 2);
            assert_eq!(bounds.max.y, 1);
        }
    }
}

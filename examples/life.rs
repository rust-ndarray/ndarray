#[macro_use]
extern crate ndarray;

use ndarray::{
    Array,
    Ix,
};

type Ix2 = (Ix, Ix);

const INPUT: &'static [u8] = include_bytes!("life.txt");
//const INPUT: &'static [u8] = include_bytes!("lifelite.txt");

const N: usize = 100;
//const N: usize = 8;

type Board = Array<u8, Ix2>;

fn parse(x: &[u8]) -> Board {
    // make a border of 0 cells
    let mut map = Array::from_elem(((N + 2) as Ix, (N + 2) as Ix), 0);
    let a: Array<u8, Ix> = x.iter().filter_map(|&b| match b {
        b'#' => Some(1),
        b'.' => Some(0),
        _ => None,
    }).collect();

    let a = a.reshape((N as Ix, N as Ix));
    map.slice_mut(s![1..-1, 1..-1]).assign(&a);
    map
}

// Rules
//
// 2 or 3 neighbors: stay alive
// 3 neighbors: birth
// otherwise: death

fn iterate(z: &mut Board, scratch: &mut Board) {
    // compute number of neighbors
    let mut neigh = scratch.view_mut();
    neigh.assign_scalar(&0);
    let neigh = neigh
    + z.slice(s![0..-2, 0..-2])
    + z.slice(s![0..-2, 1..-1])
    + z.slice(s![0..-2, 2..  ])

    + z.slice(s![1..-1, 0..-2])
    + z.slice(s![1..-1, 2..  ])

    + z.slice(s![2..  , 0..-2])
    + z.slice(s![2..  , 1..-1])
    + z.slice(s![2..  , 2..  ]);

    // birth where n = 3 and z[i] = 0,
    // survive where n = 2 || n = 3 and z[i] = 1
    {
        let mut zv = z.slice_mut(s![1..-1, 1..-1]);

        zv.zip_with_mut(&neigh, |y, &n| {
            if n == 3 {
                *y = 1;
            } else if n == 2 {
            } else {
                *y = 0;
            }
        });
    }
}

fn turn_on_corners(z: &mut Board) {
    z.slice_mut(s![1..2, 1..2]).assign_scalar(&1);
    z.slice_mut(s![1..2, -2..-1]).assign_scalar(&1);
    z.slice_mut(s![-2..-1, 1..2]).assign_scalar(&1);
    z.slice_mut(s![-2..-1, -2..-1]).assign_scalar(&1);
}

fn render(a: &Board) {
    for row in 0..a.shape()[0] {
        for &x in a.row_iter(row) {
            if x > 0 {
                print!("#");
            } else {
                print!(".");
            }
        }
        println!("");
    }
}

fn main() {
    let mut a = parse(INPUT);
    let mut scratch = Board::zeros((N as Ix, N as Ix));
    let steps = 100;
    turn_on_corners(&mut a);
    for _ in 0..steps {
        iterate(&mut a, &mut scratch);
        turn_on_corners(&mut a);
        //render(&a);
    }
    render(&a);
    let alive = a.iter().filter(|&&x| x > 0).count();
    println!("After {} steps there are {} cells alive", steps, alive);
}

%helper fucntions
function print_mtx(mtxA)
    [rows, cols] = size(mtxA);

    if rows <= 6
        row_indices = 1:rows;
    else
        row_indices = [1:3, rows-2:rows];
    end

    if cols <= 6
        col_indices = 1:cols;
    else
        col_indices = [1:3, cols-2:cols];
    end

    disp(mtxA(row_indices, col_indices));
end

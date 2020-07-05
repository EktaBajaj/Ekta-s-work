

#' curatedGene
#' @title Curate Gene data
#' @param input_data 
#' @author Ekta Bajaj
#' @return data.frame
#' @import stringr
#' @export
#' @description Use this function to map gene info
#' @examples input_data = filename
curatedGene <- function(input_data){
  input_data$result <- ifelse(grepl("-Bt.Bt.*:", input_data$input), str_extract(input_data$input,"(?<=Bt.).*([^\\:]+{2}"),
                              ifelse(grepl("-.*:", input_data$input), str_extract(input_data$input,"(?<=-).*([^\\:]+{2}"),input_data$input))
  input_data$result <- ifelse(grepl("-[0-9]$",input_data$result),gsub(".{2}$"," ", input_data$result), input_data$result)
  return(input_data)
                                                                                
}

#' CuratedInset
#' @title Curate Insect data
#' @param input_data 
#' @param std.library 
#' @author Ekta Bajaj
#' @return data.frame
#' @import stringr
#' @export
#' @description Use this function to map Insect info
#' @examples input_data = filename, std.library = insectlibrary
curatedInsect <- function(input_data, std.library){
  result <- merge(input_data, std.library, by.x = 'var1', by.y = 'var2')
  result$output <- ifelse(result$refcode=="A" | result$refcode=="B", "insect1",
                          ifelse(result$refcode=="C" & grepl(pattern = "Infestation", result$title),"insect2",
                                 ifelse(result$refcode=="D" &(result$description =="Not Applicable" | result$description=="insect3") & grepl(pattern = ".*A",result$title), "insect4",
                                        ifelse((result$refcode=="E" | result$refcode=="F") & grepl(pattern = ".*B.*",result$title), "insect4",
                                               ifelse(result$description == "Not Applicable" & (result$refcode=="G" | result$refcode=="C") & grepl(pattern = "Natural", result$title),"Unkonwn",result$output)))))
                          
  return(result)
}




#' CuratedStage
#' @title Curate Stage data
#' @param input_data 
#' @param cheatSheet 
#' @author Ekta Bajaj
#' @return data.frame
#' @import stringr
#' @export
#' @description Use this function to map Stage info
#' @examples input_data = filename, cheatSheet = sheet
curatedStgae <- function(input_data, cheatSheet) {
  result <- merge(input_data, cheatSheet, by.x = 'var1', by.y = 'var2')
  result$output <- ifelse(result$refcode=="A"|result$refcode=="B"|result$refcode=="C"|result$refcode=="D", "R4",
                          ifelse(result$refcode=="E" & grepl("V|R", result$description), str_extract(as.character(result$description),"V[A-Z0-9]{1,2}|R[A-Z0-9]{1,2}"),
                                 ifelse(result$refcode=="F" & grepl(pattern = "A|B|C", result$title), "V11", result$output)))
  return(result)
}
